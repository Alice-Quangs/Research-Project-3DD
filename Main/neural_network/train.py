"""Train a separate weighted NN for HA1, HA2 or HA3 using shared data.

Run from repository root:
 python -m neural_network.train --heuristic ha1

Requires numpy, pandas, joblib, scikit-learn with MLPClassifier sample_weight
support (>=1.7). Uses the legacy logistic/L-BFGS architecture search: one
hidden layer with 1,3,5,7,9 units and alpha 0,.0001,.001,.01. Select maximum
UNWEIGHTED validation average precision; ties retain the first candidate.
No row-level CV, undersampling, internal random validation split or test read.
L-BFGS uses its convergence criterion/max_iter, not validation early stopping.
Convergence warnings are recorded in tuning.csv and shown during training.

StandardScaler fits TRAIN rows only. Positive training weight = Nneg/Npos;
negative weight = 1. Validation remains naturally distributed/unweighted.
The selected fitted candidate is retained (no train+validation refit). Its
threshold is selected on validation by metrics.select_threshold. Filter-start
selection is a separate packing experiment with benchmark.py.

Input defaults: data/processed/HEURISTIC/train.csv and validation.csv, with
.meta.json sidecars from generate_dataset.py, plus the saved split manifest.
All 19 features are used in config.py order. Order IDs retain leading zeros.
CSV rows are loaded into memory; ensure sufficient RAM for the full dataset.

Output NEW directory: models/HEURISTIC/neural_network/
 model.joblib (dict with model, scaler, threshold, feature schema and metadata),
 metadata.json, tuning.csv, validation_metrics.json, threshold_sweep.json,
 validation_predictions.csv. No test metrics or feature importance are computed
here. The forthcoming predictor.py must load this bundle's model/scaler.
Do not load untrusted joblib files. Saved models depend on package versions.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from config import CONFIG, FEATURE_SCHEMA_VERSION, get_feature_names
from bin_types import bin_types
from metrics import select_threshold


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


from training_data import load_partition


def fit_network(X_train,y_train,X_validation,y_validation,*,hidden_sizes=(1,3,5,7,9),
                alphas=(0.,.0001,.001,.01),max_iter=1000,max_fun=15000,seed=42):
    """Fit candidates with shared weights; return bundle parts and tuning records."""
    if 'sample_weight' not in inspect.signature(MLPClassifier.fit).parameters:
        raise RuntimeError('Install scikit-learn >=1.7: this version cannot weight MLP training.')
    if (CONFIG.class_weights.policy!='scale_weighted' or CONFIG.class_weights.undersample
            or CONFIG.class_weights.negative_weight!=1
            or CONFIG.class_weights.positive_weight_rule!='n_negative_divided_by_n_positive'):
        raise ValueError('Unsupported class-weight configuration.')
    if not hidden_sizes or any(type(n) is not int or n<1 for n in hidden_sizes):
        raise ValueError('Hidden sizes must be positive integers.')
    if not alphas or any(not np.isfinite(a) or a<0 for a in alphas):
        raise ValueError('Alphas must be finite and nonnegative.')
    for X,y in ((X_train,y_train),(X_validation,y_validation)):
        if X.ndim!=2 or len(X)!=len(y) or not np.isfinite(X).all() or set(np.unique(y))!={0,1}:
            raise ValueError('Require finite feature matrices and both binary classes.')
    if X_train.shape[1]!=X_validation.shape[1]:raise ValueError('Feature counts differ.')
    ratio=float(np.sum(y_train==0)/np.sum(y_train==1))
    weights=np.where(y_train==1,ratio,1.)
    scaler=StandardScaler().fit(X_train)
    train=scaler.transform(X_train);validation=scaler.transform(X_validation)
    best=None;best_score=-np.inf;records=[]
    for size in hidden_sizes:
        for alpha in alphas:
            model=MLPClassifier(hidden_layer_sizes=(size,),activation='logistic',solver='lbfgs',
                                alpha=alpha,max_iter=max_iter,max_fun=max_fun,random_state=seed)
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(train,y_train,sample_weight=weights)
            elapsed=time.perf_counter()-start
            probability=model.predict_proba(validation)[:,list(model.classes_).index(1)]
            if not np.isfinite(probability).all():raise RuntimeError('Nonfinite validation predictions.')
            score=float(average_precision_score(y_validation,probability))
            notes=[str(w.message) for w in caught]
            record=dict(hidden_units=size,alpha=alpha,validation_average_precision=score,
                        fit_seconds=elapsed,iterations=int(model.n_iter_),loss=float(model.loss_),
                        convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught),
                        warnings=' | '.join(notes))
            records.append(record)
            print(f'Hidden={size}, alpha={alpha:g}: validation AP={score:.6f}; warnings={len(notes)}',flush=True)
            if score>best_score:
                best_score=score;best=(model,probability,dict(record))
    return best[0],scaler,best[1],best[2],records,ratio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heuristic',choices=('ha1','ha2','ha3'),default=CONFIG.heuristic)
    parser.add_argument('--train',type=Path)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--split-manifest',type=Path,default=CONFIG.paths.split_manifest)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--hidden-sizes',type=int,nargs='+',default=[1,3,5,7,9])
    parser.add_argument('--alphas',type=float,nargs='+',default=[0.,.0001,.001,.01])
    parser.add_argument('--max-iter',type=int,default=1000)
    parser.add_argument('--max-fun',type=int,default=15000)
    args=parser.parse_args()
    CONFIG.validate()
    output=args.output or CONFIG.paths.models_dir/args.heuristic/'neural_network'
    if output.exists():parser.error('Output already exists; choose a new experiment directory.')
    if args.max_iter<1 or args.max_fun<1:parser.error('Iteration limits must be positive.')
    train_path=args.train or CONFIG.paths.processed_dir/args.heuristic/'train.csv'
    validation_path=args.validation or CONFIG.paths.processed_dir/args.heuristic/'validation.csv'
    manifest=json.loads(args.split_manifest.read_text())
    seen=set()
    for partition in ('train','validation','test'):
        ids=manifest.get(partition)
        if not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids):
            raise ValueError('Manifest needs nonempty string ID lists for every partition.')
        if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Split IDs overlap or repeat.')
        seen.update(ids)
    X,y,_,tm=load_partition(train_path,'train',args.heuristic,manifest)
    V,vy,tracking,vm=load_partition(validation_path,'validation',args.heuristic,manifest)
    if tm.get('label_source')!=vm.get('label_source'):raise ValueError('Train/validation label definitions differ.')
    model,scaler,p,best,records,ratio=fit_network(X,y,V,vy,hidden_sizes=args.hidden_sizes,
        alphas=args.alphas,max_iter=args.max_iter,max_fun=args.max_fun,seed=CONFIG.data.random_seed)
    selection=select_threshold(vy,p)
    metadata=dict(artifact_version=1,model_type='neural_network',heuristic=args.heuristic,
                  feature_names=list(get_feature_names(args.heuristic)),feature_schema_version=FEATURE_SCHEMA_VERSION,
                  threshold=selection['threshold'],threshold_rule=CONFIG.thresholds.selection_method,
                  target_recall=CONFIG.thresholds.target_recall,selected_min_bins=None,
                  training_rows=len(y),validation_rows=len(vy),positive_weight=ratio,negative_weight=1.,
                  train_positive=int(y.sum()),train_negative=int((y==0).sum()),
                  selection_metric='validation_average_precision',best_candidate=best,
                  training_parameters=model.get_params(),label_source=tm.get('label_source'),bin_catalogue=bin_types,
                  train_csv_sha256=_hash(train_path),validation_csv_sha256=_hash(validation_path),
                  split_manifest_sha256=_hash(args.split_manifest),seed=CONFIG.data.random_seed,
                  sklearn_version=sklearn.__version__,numpy_version=np.__version__)
    bundle=dict(metadata,model=model,scaler=scaler)
    output.mkdir(parents=True,exist_ok=False)
    joblib.dump(bundle,output/'model.joblib',compress=3)
    pd.DataFrame(records).to_csv(output/'tuning.csv',index=False)
    tracking=tracking.copy();tracking['fit']=vy;tracking['probability']=p
    tracking['predicted_fit']=(p>=selection['threshold']).astype(int)
    tracking.to_csv(output/'validation_predictions.csv',index=False)
    for name,content in [('metadata',metadata),('validation_metrics',selection['metrics']),
                         ('threshold_sweep',selection)]:
        (output/f'{name}.json').write_text(json.dumps(content,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Saved {output}; threshold={selection["threshold"]:.4f}. Test data was not used.')


if __name__=='__main__':
    main()
