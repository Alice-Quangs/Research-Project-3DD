"""NN inference adapter shared by HA1, HA2 and HA3.

Load trusted bundles created by neural_network/train.py:
    predictor = load_predictor('models/ha1/neural_network/model.joblib')
    probabilities = predictor.predict_fit_probability(X)

X contains RAW shared features in config.py order. Scaling is applied here
exactly once. Output shape is (n,), P(fit=1). Thresholding, filter activation
and fallback remain in filtering.py. Empty input shape (0,19) is supported.

Fast inference: saved StandardScaler parameters + NumPy dense layers with
SciPy's stable expit (logistic activation). Native inference: scaler.transform
then the original sklearn MLPClassifier.predict_proba. Native bypasses caches.
A bounded LRU cache keys on complete float64 feature-row bytes, never object
identity; changed bin-state features therefore produce a new key. Each loaded
adapter owns a fresh cache. clear_cache() resets it between experiments.

validate_fast=True checks deterministic synthetic raw probes at load time.
This is only a smoke check, NOT validation on real candidate states. Run
scripts/validate_inference.py on saved validation rows before benchmarking.
Validation rejects threshold decision differences even within tolerance.

Dependencies: numpy, scipy, scikit-learn, joblib. Python 3.10+.
Only load trusted joblib files: deserialisation can execute arbitrary code.
Use the scikit-learn version recorded by the training bundle.
"""
from collections import OrderedDict
from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import CONFIG, FEATURE_SCHEMA_VERSION, get_feature_names
from bin_types import bin_types


class NeuralNetworkPredictor:
    def __init__(self,bundle,*,cache_enabled=True,cache_size=50000):
        if not isinstance(bundle,dict) or bundle.get('artifact_version')!=1:
            raise ValueError('Expected a version-1 bundle from neural_network/train.py.')
        if bundle.get('model_type')!='neural_network':raise ValueError('Not an NN bundle.')
        self.heuristic=bundle.get('heuristic')
        self.feature_names=tuple(get_feature_names(self.heuristic))
        if (tuple(bundle.get('feature_names',()))!=self.feature_names
                or bundle.get('feature_schema_version')!=FEATURE_SCHEMA_VERSION
                or bundle.get('bin_catalogue')!=bin_types):
            raise ValueError('Model feature schema or bin catalogue differs from this framework.')
        self.model=bundle.get('model');self.scaler=bundle.get('scaler')
        if not isinstance(self.model,MLPClassifier) or not isinstance(self.scaler,StandardScaler):
            raise ValueError('Bundle must contain fitted MLPClassifier and StandardScaler objects.')
        if (self.model.activation!='logistic' or getattr(self.model,'out_activation_',None)!='logistic'
                or not np.array_equal(getattr(self.model,'classes_',None),[0,1])):
            raise ValueError('Fast path requires logistic hidden/output activation and binary classes [0,1].')
        n=len(self.feature_names)
        if getattr(self.model,'n_features_in_',None)!=n or getattr(self.scaler,'n_features_in_',None)!=n:
            raise ValueError('Fitted model/scaler feature count mismatch.')
        if not self.scaler.with_mean or not self.scaler.with_std:
            raise ValueError('Expected training pipeline StandardScaler with mean/std scaling.')
        self.mean=np.array(self.scaler.mean_,dtype=float,copy=True)
        self.scale=np.array(self.scaler.scale_,dtype=float,copy=True)
        self.weights=[np.array(w,dtype=float,copy=True) for w in self.model.coefs_]
        self.biases=[np.array(b,dtype=float,copy=True) for b in self.model.intercepts_]
        if self.mean.shape!=(n,) or self.scale.shape!=(n,) or np.any(self.scale<=0):
            raise ValueError('Invalid scaler parameters.')
        if not self.weights or len(self.weights)!=len(self.biases):raise ValueError('Invalid layers.')
        previous=n
        for w,b in zip(self.weights,self.biases):
            if w.ndim!=2 or w.shape[0]!=previous or b.shape!=(w.shape[1],):
                raise ValueError('Invalid network parameter shapes.')
            previous=w.shape[1]
        if previous!=1:raise ValueError('Expected one binary output unit.')
        if not all(np.isfinite(a).all() for a in [self.mean,self.scale,*self.weights,*self.biases]):
            raise ValueError('Nonfinite fitted parameters.')
        self.threshold=float(bundle['threshold'])
        if not np.isfinite(self.threshold) or not 0<=self.threshold<=1:raise ValueError('Invalid threshold.')
        if type(cache_size) is not int or cache_size<1:raise ValueError('cache_size must be positive.')
        self.cache_enabled=bool(cache_enabled);self.cache_size=cache_size
        self._cache=OrderedDict();self.cache_hits=0;self.cache_misses=0
        self.validation_report=None
        trained_version=bundle.get('sklearn_version')
        if trained_version!=sklearn.__version__:
            warnings.warn(f'Model trained with scikit-learn {trained_version}; running {sklearn.__version__}.',RuntimeWarning)

    def clear_cache(self):
        self._cache.clear();self.cache_hits=0;self.cache_misses=0

    def _input(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[1]!=len(self.feature_names) or not np.isfinite(X).all():
            raise ValueError('Expected a finite raw feature matrix of shape (n,19).')
        return X

    def _forward(self,X):
        activation=(X-self.mean)/self.scale
        for weights,bias in zip(self.weights,self.biases):
            activation=expit(activation @ weights+bias)
        result=activation[:,0]
        if not np.isfinite(result).all():raise ValueError('Nonfinite fast probabilities.')
        return result

    def predict_fit_probability(self,X):
        X=self._input(X)
        if not len(X):return np.empty(0,dtype=float)
        if not self.cache_enabled:return self._forward(X)
        result=np.empty(len(X),dtype=float)
        missing={}
        for index,row in enumerate(X):
            key=row.tobytes()
            if key in self._cache:
                result[index]=self._cache[key];self._cache.move_to_end(key);self.cache_hits+=1
            else:
                missing.setdefault(key,[]).append(index);self.cache_misses+=1
        if missing:
            keys=list(missing)
            values=self._forward(X[[missing[key][0] for key in keys]])
            for key,value in zip(keys,values):
                result[missing[key]]=value
                self._cache[key]=float(value);self._cache.move_to_end(key)
                if len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        return result

    def predict_native_fit_probability(self,X):
        X=self._input(X)
        if not len(X):return np.empty(0,dtype=float)
        return self.model.predict_proba(self.scaler.transform(X))[:,1]

    def validate(self,X,*,atol=1e-6,rtol=1e-5):
        """Compare uncached fast/native outputs on caller-supplied raw rows."""
        X=self._input(X)
        if not len(X):raise ValueError('Validation requires at least one row.')
        fast=self._forward(X);native=self.predict_native_fit_probability(X)
        errors=np.abs(fast-native)
        mismatch=int(np.sum((fast>=self.threshold)!=(native>=self.threshold)))
        report=dict(rows=len(X),max_abs_error=float(errors.max()),
                    outside_tolerance=int(np.sum(errors>atol+rtol*np.abs(native))),
                    threshold=self.threshold,decision_mismatches=mismatch)
        report['passed']=report['outside_tolerance']==0 and mismatch==0
        if not report['passed']:raise ValueError(f'Fast/native inference validation failed: {report}')
        return report


def load_predictor(path,*,cache_enabled=CONFIG.benchmark.probability_cache_enabled,
                   validate_fast=True,cache_size=50000):
    """Return a fresh adapter; smoke validation never populates its cache."""
    predictor=NeuralNetworkPredictor(joblib.load(Path(path)),cache_enabled=cache_enabled,cache_size=cache_size)
    if validate_fast:
        rng=np.random.default_rng(42)
        probes=predictor.mean+predictor.scale*rng.normal(size=(64,len(predictor.feature_names)))
        probes[0]=predictor.mean
        predictor.validation_report=predictor.validate(probes)
        predictor.validation_report['scope']='synthetic load-time smoke check only'
    return predictor
