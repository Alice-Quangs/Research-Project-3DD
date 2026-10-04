"""XGBoost predictor adapter for the shared HA1/HA2/HA3 framework.

Load trusted model.joblib bundles from xgb_model/train.py:
    predictor = load_predictor('models/ha1/xgb_model/model.joblib')
    probabilities = predictor.predict_fit_probability(X)

Fast path uses Booster.inplace_predict on contiguous float32 raw features.
Reference path uses XGBClassifier.predict_proba with DMatrix fallback forced
in a separate copy. This makes the comparison exercise two prediction APIs
rather than predict_proba silently choosing inplace_predict itself.
Both return P(fit=1) in shape (n,); no scaling or thresholding is performed.
The saved model may contain ANY selected tree count, including 200 trees.
This adapter requires the fixed-round binary gbtree model trained by train.py;
models with early-stopping metadata are rejected to prevent tree-range ambiguity.

Cache: bounded per-instance LRU keyed by full float32 feature-row values.
Changed input values trigger new keys; float32-equivalent rows intentionally
share keys, matching XGBoost input precision. Caches start empty at every load.

validate_fast=True performs deterministic synthetic probes ONLY; run
scripts/validate_inference.py on real validation rows before benchmarking.
No claim of speedup is made: measure end-to-end runtime with benchmark.py.

Dependencies: xgboost >=2.0, numpy, joblib and scikit-learn.
Only load trusted joblib artifacts. Use the versions recorded during training.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import warnings

import joblib
import numpy as np
import xgboost as xgb

from config import CONFIG, FEATURE_SCHEMA_VERSION, get_feature_names
from bin_types import bin_types


class XGBoostPredictor:
    def __init__(self,bundle,*,cache_enabled=True,cache_size=50000):
        if not isinstance(bundle,dict) or bundle.get('artifact_version')!=1:
            raise ValueError('Expected a version-1 bundle from xgb_model/train.py.')
        if bundle.get('model_type')!='xgboost':raise ValueError('Not an XGBoost bundle.')
        self.heuristic=bundle.get('heuristic')
        self.feature_names=tuple(get_feature_names(self.heuristic))
        if (tuple(bundle.get('feature_names',()))!=self.feature_names
                or bundle.get('feature_schema_version')!=FEATURE_SCHEMA_VERSION
                or bundle.get('bin_catalogue')!=bin_types):
            raise ValueError('Model feature schema or bin catalogue mismatch.')
        self.model=bundle.get('model')
        if not isinstance(self.model,xgb.XGBClassifier):
            raise ValueError('Bundle must contain a fitted XGBClassifier.')
        params=self.model.get_xgb_params()
        if params.get('objective')!='binary:logistic' or params.get('booster') not in (None,'gbtree'):
            raise ValueError('Expected binary:logistic gbtree model.')
        if not np.array_equal(self.model.classes_,[0,1]):raise ValueError('Expected classes [0,1].')
        self.booster=self.model.get_booster()
        if self.booster.num_features()!=len(self.feature_names):raise ValueError('Feature count mismatch.')
        if self.booster.attr('best_iteration') is not None:
            raise ValueError('Early-stopped models require an explicit tree-range adapter.')
        self.boosted_rounds=int(self.booster.num_boosted_rounds())
        expected=bundle.get('best_candidate',{}).get('boosted_rounds')
        if expected is not None and expected!=self.boosted_rounds:
            raise ValueError('Saved boosted-round count does not match model.')
        self.threshold=float(bundle['threshold'])
        if not np.isfinite(self.threshold) or not 0<=self.threshold<=1:raise ValueError('Invalid threshold.')
        if type(cache_size) is not int or cache_size<1:raise ValueError('cache_size must be positive.')
        self.cache_enabled=bool(cache_enabled);self.cache_size=cache_size
        self._cache=OrderedDict();self.cache_hits=0;self.cache_misses=0
        # A separate copy keeps reference-path control out of the live model.
        self._native=deepcopy(self.model)
        if not callable(getattr(self._native,'_can_use_inplace_predict',None)):
            raise RuntimeError('Unsupported XGBoost API: cannot force independent native path.')
        self._native._can_use_inplace_predict=lambda:False
        self.validation_report=None
        if bundle.get('xgboost_version')!=xgb.__version__:
            warnings.warn(f"Model trained with XGBoost {bundle.get('xgboost_version')}; running {xgb.__version__}.",RuntimeWarning)

    def clear_cache(self):
        self._cache.clear();self.cache_hits=0;self.cache_misses=0

    def _input(self,X):
        raw=np.asarray(X)
        if raw.ndim!=2 or raw.shape[1]!=len(self.feature_names) or raw.dtype.kind not in 'fiu':
            raise ValueError('Expected numeric raw matrix of shape (n,19).')
        if not np.isfinite(raw).all():raise ValueError('Input contains nonfinite values.')
        with np.errstate(over='ignore',invalid='ignore'):
            converted=np.ascontiguousarray(raw,dtype=np.float32)
        if not np.isfinite(converted).all():raise ValueError('Input exceeds float32 range.')
        return converted

    @staticmethod
    def _output(p,n):
        p=np.asarray(p,dtype=float)
        if p.shape!=(n,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid prediction probabilities.')
        return p

    def _fast(self,X):
        return self._output(self.booster.inplace_predict(X,iteration_range=(0,0),
                            predict_type='value',validate_features=True),len(X))

    def predict_fit_probability(self,X):
        X=self._input(X)
        if not len(X):return np.empty(0,dtype=float)
        if not self.cache_enabled:return self._fast(X)
        result=np.empty(len(X),dtype=float);missing={}
        for index,row in enumerate(X):
            key=row.tobytes()
            if key in self._cache:
                result[index]=self._cache[key];self._cache.move_to_end(key);self.cache_hits+=1
            else:
                missing.setdefault(key,[]).append(index);self.cache_misses+=1
        if missing:
            keys=list(missing)
            p=self._fast(X[[missing[key][0] for key in keys]])
            for key,value in zip(keys,p):
                result[missing[key]]=value;self._cache[key]=float(value)
                self._cache.move_to_end(key)
                if len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        return result

    def predict_native_fit_probability(self,X):
        X=self._input(X)
        if not len(X):return np.empty(0,dtype=float)
        return self._output(self._native.predict_proba(X,iteration_range=(0,0))[:,1],len(X))

    def validate(self,X,*,atol=1e-6,rtol=1e-5):
        X=self._input(X)
        if not len(X):raise ValueError('Validation needs at least one row.')
        fast=self._fast(X);native=self.predict_native_fit_probability(X)
        errors=np.abs(fast-native)
        report=dict(rows=len(X),max_abs_error=float(errors.max()),
                    outside_tolerance=int(np.sum(errors>atol+rtol*np.abs(native))),
                    threshold=self.threshold,
                    decision_mismatches=int(np.sum((fast>=self.threshold)!=(native>=self.threshold))))
        report['passed']=report['outside_tolerance']==0 and report['decision_mismatches']==0
        if not report['passed']:raise ValueError(f'Inference validation failed: {report}')
        return report


def load_predictor(path,*,cache_enabled=CONFIG.benchmark.probability_cache_enabled,
                   validate_fast=True,cache_size=50000):
    predictor=XGBoostPredictor(joblib.load(Path(path)),cache_enabled=cache_enabled,cache_size=cache_size)
    if validate_fast:
        # Broad synthetic magnitudes; these are not physically valid packing states.
        rng=np.random.default_rng(42)
        probes=rng.uniform(0,1,size=(64,len(predictor.feature_names)))
        for index,name in enumerate(predictor.feature_names):
            if 'volume' in name and name!='item_to_bin_volume_ratio':probes[:,index]*=1e8
            elif any(term in name for term in ('length','width','height','eps_')):probes[:,index]*=1000
        probes[0]=0
        predictor.validation_report=predictor.validate(probes)
        predictor.validation_report['scope']='synthetic load-time smoke check only'
    return predictor
