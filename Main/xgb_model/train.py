"""Train one weighted XGBoost model per heuristic using the shared pipeline.

Run from repository root: python -m xgb_model.train --heuristic ha1
Requires xgboost >=2.0, numpy, pandas, joblib, scikit-learn and training_data.py.

Default proposed grid: depths 2,3,4,5 crossed with 10,25,50,100,200 trees
(20 candidates, matching the NN candidate count). Learning rate .08 follows
the earlier retraining script. These are candidate settings, not established
optimal values. One CPU thread by default; change --threads explicitly.

Fit all training rows without scaling/undersampling. scale_pos_weight=Nneg/Npos
uses training labels only. Validation rows are unweighted. Select maximum
validation average precision, retaining the first candidate on exact ties.
This fixed-tree-count search uses NO early stopping so tested tree counts
are exact. Do not refit on validation. Shared recall-constrained threshold
selection is performed afterwards. Test rows are never read.

Output NEW directory: models/HEURISTIC/xgb_model/ containing model.joblib,
booster.ubj, metadata.json, tuning.csv, validation_metrics.json,
threshold_sweep.json and validation_predictions.csv. Predictor.py must load
the saved bundle, respecting feature_names, heuristic and threshold. No scaler
is present. booster.ubj is an additional native model, not a complete bundle.

Use saved train.csv and validation.csv plus .meta.json sidecars from the shared
generator and split manifest. Full CSVs are loaded into RAM. Generated-row
checks are shared with NN via training_data.py. Classifier selection does not
select the filter-start setting or guarantee the fastest packing configuration;
run the separate validation packing benchmark next.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score
from config import CONFIG, FEATURE_SCHEMA_VERSION, get_feature_names
from bin_types import bin_types
from metrics import select_threshold
from training_data import load_partition


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fit_xgboost(X_train,y_train,X_validation,y_validation,*,depths=(2,3,4,5),
                trees=(10,25,50,100,200),learning_rate=.08,threads=1,seed=42):
    if (CONFIG.class_weights.policy!='scale_weighted' or CONFIG.class_weights.undersample
            or CONFIG.class_weights.negative_weight!=1
            or CONFIG.class_weights.positive_weight_rule!='n_negative_divided_by_n_positive'):
        raise ValueError('Unsupported class-weight configuration.')
    if not depths or not trees or any(type(n) is not int or n<1 for n in (*depths,*trees)):
        raise ValueError('Depths and tree counts must be positive integers.')
    if not np.isfinite(learning_rate) or not 0<learning_rate<=1:
        raise ValueError('learning_rate must be in (0,1].')
    if type(threads) is not int or threads<1:raise ValueError('threads must be positive.')
    for X,y in ((X_train,y_train),(X_validation,y_validation)):
        if X.ndim!=2 or len(X)!=len(y) or not np.isfinite(X).all() or set(np.unique(y))!={0,1}:
            raise ValueError('Require finite feature matrices and both binary classes.')
    if X_train.shape[1]!=X_validation.shape[1]:raise ValueError('Feature counts differ.')
    ratio=float(np.sum(y_train==0)/np.sum(y_train==1))
    best=None;best_score=-np.inf;records=[]
    for depth in depths:
        for count in trees:
            model=xgb.XGBClassifier(objective='binary:logistic',tree_method='hist',
                n_estimators=count,max_depth=depth,learning_rate=learning_rate,
                scale_pos_weight=ratio,subsample=1.,colsample_bytree=1.,
                min_child_weight=1.,reg_alpha=0.,reg_lambda=1.,
                n_jobs=threads,random_state=seed,eval_metric='logloss')
            start=time.perf_counter()
            model.fit(X_train,y_train)
            elapsed=time.perf_counter()-start
            p=model.predict_proba(X_validation)[:,list(model.classes_).index(1)]
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite validation predictions.')
            score=float(average_precision_score(y_validation,p))
            record=dict(max_depth=depth,n_estimators=count,validation_average_precision=score,
                        fit_seconds=elapsed,boosted_rounds=int(model.get_booster().num_boosted_rounds()))
            records.append(record)
            print(f'Depth={depth}, trees={count}: validation AP={score:.6f}',flush=True)
            if score>best_score:best_score=score;best=(model,p,dict(record))
    return best[0],best[1],best[2],records,ratio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heuristic',choices=('ha1','ha2','ha3'),default=CONFIG.heuristic)
    parser.add_argument('--train',type=Path)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--split-manifest',type=Path,default=CONFIG.paths.split_manifest)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--depths',type=int,nargs='+',default=[2,3,4,5])
    parser.add_argument('--trees',type=int,nargs='+',default=[10,25,50,100,200])
    parser.add_argument('--learning-rate',type=float,default=.08)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    CONFIG.validate()
    output=args.output or CONFIG.paths.models_dir/args.heuristic/'xgb_model'
    if output.exists():parser.error('Output already exists; choose a new experiment directory.')
    if args.threads<1:parser.error('threads must be positive.')
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
    model,p,best,records,ratio=fit_xgboost(X,y,V,vy,depths=args.depths,trees=args.trees,
        learning_rate=args.learning_rate,threads=args.threads,seed=CONFIG.data.random_seed)
    selection=select_threshold(vy,p)
    metadata=dict(artifact_version=1,model_type='xgboost',heuristic=args.heuristic,
                  feature_names=list(get_feature_names(args.heuristic)),feature_schema_version=FEATURE_SCHEMA_VERSION,
                  threshold=selection['threshold'],threshold_rule=CONFIG.thresholds.selection_method,
                  target_recall=CONFIG.thresholds.target_recall,selected_min_bins=None,
                  training_rows=len(y),validation_rows=len(vy),positive_weight=ratio,negative_weight=1.,
                  train_positive=int(y.sum()),train_negative=int((y==0).sum()),
                  selection_metric='validation_average_precision',best_candidate=best,
                  training_parameters=model.get_params(),label_source=tm.get('label_source'),bin_catalogue=bin_types,
                  train_csv_sha256=_hash(train_path),validation_csv_sha256=_hash(validation_path),
                  split_manifest_sha256=_hash(args.split_manifest),seed=CONFIG.data.random_seed,
                  sklearn_version=sklearn.__version__,numpy_version=np.__version__,xgboost_version=xgb.__version__)
    bundle=dict(metadata,model=model)
    output.mkdir(parents=True,exist_ok=False)
    joblib.dump(bundle,output/'model.joblib',compress=3)
    model.save_model(output/'booster.ubj')
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
