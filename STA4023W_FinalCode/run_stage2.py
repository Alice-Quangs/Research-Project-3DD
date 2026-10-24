"""Run from Main. Uses existing benchmark/config/loader/split/predictor interfaces.
See EXPERIMENT_GUIDE.md. No training or settings selection occurs in this runner.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import time
from contextlib import contextmanager


def write_csv(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def combine(orders, targets, count, seed):
    """Whole-order concatenation, no source reused within one synthetic instance.
    Sources CAN recur across synthetic instances: report the dependence.
    """
    rng=random.Random(seed); output=[]; manifest=[]
    for target in targets:
        if sum(len(s) for _,s in orders)<target: raise ValueError('Insufficient source items.')
        for instance in range(count):
            indices=list(range(len(orders)));rng.shuffle(indices)
            seq=[];ids=[]
            for i in indices:
                oid,source=orders[i];ids.append(oid)
                seq.extend(dict(item) for item in source)
                if len(seq)>=target: break
            oid=f'combined_{target}_{instance:04d}'
            output.append((oid,seq));manifest.append(dict(order_id=oid,target_items=target,actual_items=len(seq),source_order_ids=ids))
    return output,manifest


@contextmanager
def diagnostics(module):
    """Intrusive diagnostic counters; NEVER use these runtimes as headline timings.
    geometry_primitive_calls = orientation/EMS or orientation/EP tests, not pairs
    checked inside EP collision loops. pack_call_s includes placement/state updates.
    feature_s and prediction_s are subcomponents of filter_s, not additive to it.
    """
    import filtering
    stats={k:0 for k in ['geometry_primitive_calls','pack_calls','feature_s','prediction_s','filter_s','pack_call_s']}
    original=[]
    def wrap(obj,name,key,timed=True):
        old=getattr(obj,name)
        def f(*a,**kw):
            if not timed:
                stats[key]+=1;return old(*a,**kw)
            t=time.perf_counter()
            try: return old(*a,**kw)
            finally: stats[key]+=time.perf_counter()-t
        original.append((obj,name,old));setattr(obj,name,f)
    wrap(module,'fits_in_ems' if hasattr(module,'fits_in_ems') else 'can_place','geometry_primitive_calls',False)
    name='pack_item_unfiltered' if hasattr(module,'pack_item_unfiltered') else 'pack_item'
    wrap(module,name,'pack_calls',False);wrap(module,name,'pack_call_s')
    wrap(filtering,'build_feature_matrix','feature_s');wrap(filtering,'filter_candidates','filter_s')
    try: yield stats
    finally:
        for obj,name,old in reversed(original):setattr(obj,name,old)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heuristic',required=True,choices=['ha1','ha2','ha3'])
    p.add_argument('--mode',required=True,choices=['baseline','validation','test','combined','diagnostic'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-bins',type=int)
    p.add_argument('--repetitions',type=int,default=7)
    p.add_argument('--models-root',type=Path,default=Path('models'))
    p.add_argument('--dataset',type=Path)
    p.add_argument('--split-manifest',type=Path)
    p.add_argument('--sizes',type=int,nargs='+',default=[100,200,400,800])
    p.add_argument('--instances',type=int,default=30)
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if args.output.exists(): p.error('Choose a new output folder; existing results are never overwritten.')
    if args.repetitions<1 or args.instances<1 or any(s<1 for s in args.sizes):p.error('Counts must be positive.')
    if args.mode in ['test','combined','diagnostic'] and (args.min_bins is None or args.min_bins<1):p.error('Supply the fixed validation-selected --min-bins.')
    from config import CONFIG
    from data_loader import load_orders
    from splits import ensure_split_manifest,select_orders,dataset_fingerprint
    from benchmark import Method,benchmark_orders
    from bin_types import bin_types
    dataset=args.dataset or CONFIG.paths.raw_dataset
    split_path=args.split_manifest or CONFIG.paths.split_manifest
    if not split_path.is_file():p.error('Existing split manifest required; do not regenerate a different split.')
    all_orders=load_orders(dataset); split=ensure_split_manifest(all_orders,split_path)
    partition='validation' if args.mode in ['baseline','validation','diagnostic'] else 'test'
    orders=select_orders(all_orders,split,partition)
    construction=[]
    if args.mode=='combined':orders,construction=combine(orders,args.sizes,args.instances,args.seed)
    module=importlib.import_module('heuristics.'+args.heuristic)
    minima=[1,10,20,30,40] if args.mode=='validation' else [args.min_bins or 1]
    methods=[Method('baseline')];artifacts={}
    if args.mode!='baseline':
        for method,folder in [('xgboost','xgb_model'),('neural_network','neural_network')]:
            directory=args.models_root/args.heuristic/folder
            path=directory/'model.joblib';meta=json.loads((directory/'metadata.json').read_text())
            threshold=meta['threshold'];loader=importlib.import_module(folder+'.predictor').load_predictor
            def factory(loader=loader,path=path):
                return loader(path,cache_enabled=CONFIG.benchmark.probability_cache_enabled,validate_fast=CONFIG.benchmark.validate_fast_predictions)
            methods.append(Method(method,factory,threshold))
            artifacts[method]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),threshold=threshold)
    args.output.mkdir(parents=True)
    meta=dict(heuristic=args.heuristic,mode=args.mode,partition=partition,repetitions=args.repetitions,min_bins=minima,models=artifacts,seed=args.seed,bin_catalogue=bin_types,orders_fingerprint=dataset_fingerprint(orders),split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),heuristic_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),python=sys.version,config=CONFIG.to_dict(),timing='Inherited benchmark cold-start packing-loop timing; loading/reporting excluded.',synthetic_sources_reused_across_instances=bool(construction))
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    if construction:(args.output/'construction.json').write_text(json.dumps(construction,indent=2))
    diag=[]
    if args.mode=='diagnostic':
        rows=[];runs=[]
        # Same rotation of method order across repetitions as main benchmark.
        for rep in range(1,args.repetitions+1):
            offset=(rep-1)%len(methods)
            for method in methods[offset:]+methods[:offset]:
                with diagnostics(module) as stats:
                    selected=method
                    if method.predictor_factory:
                        base_factory=method.predictor_factory
                        def profiled_factory(base_factory=base_factory,stats=stats):
                            predictor=base_factory();old=predictor.predict_fit_probability
                            def predict(X):
                                start=time.perf_counter()
                                try:return old(X)
                                finally:stats['prediction_s']+=time.perf_counter()-start
                            class TimedPredictor:
                                def __getattr__(self, name):
                                    return getattr(predictor, name)
                                def predict_fit_probability(self, X):
                                    return predict(X)
                            return TimedPredictor()
                        selected=Method(method.name,profiled_factory,method.threshold)
                    # API requires a baseline: discard the extra baseline pass
                    # and reset its counters at model construction.
                    if method.name!='baseline':
                        inner=selected.predictor_factory
                        def reset_factory(inner=inner,stats=stats):
                            predictor=inner()
                            for key in stats:stats[key]=0
                            return predictor
                        selected=Method(selected.name,reset_factory,selected.threshold)
                    r,u,_=benchmark_orders(orders,module,[Method('baseline'),selected] if method.name!='baseline' else [selected],heuristic=args.heuristic,min_bins_values=minima,repetitions=1,partition=partition)
                    for row in r:
                        if row['method']==method.name:row['repetition']=rep;rows.append(row)
                    for row in u:
                        if row['method']==method.name:row['repetition']=rep;runs.append(row)
                    diag.append(dict(method=method.name,repetition=rep,**stats))
        write_csv(args.output/'diagnostics.csv',diag)
    else:
        rows,runs,_=benchmark_orders(orders,module,methods,heuristic=args.heuristic,min_bins_values=minima,repetitions=args.repetitions,partition=partition)
    write_csv(args.output/'per_order.csv',rows);write_csv(args.output/'per_run.csv',runs)
    import pandas as pd
    r=pd.DataFrame(runs);r['min_bins']=r.min_bins.fillna(0)
    r['activation_pct']=100*r.filter_activations/r['items']
    r['candidate_reduction_pct']=100*(1-r.shortlisted_candidates/r.available_candidates.replace(0,float('nan')))
    r['fallback_pct_of_activations']=100*r.empty_shortlist_fallbacks/r.filter_activations.replace(0,float('nan'))
    summary=r.groupby(['method','min_bins'])[['runtime_s','cost','bins','utilisation','activation_pct','candidate_reduction_pct','fallback_pct_of_activations']].mean().reset_index()
    b=summary[summary.method=='baseline'].iloc[0]
    summary['speedup']=b.runtime_s/summary.runtime_s
    summary['cost_change_pct']=100*(summary.cost/b.cost-1)
    summary.to_csv(args.output/'mean_summary.csv',index=False)
    print(summary.to_string(index=False))
    if args.mode in ['test','combined','baseline']:
        from order_size_analysis import analyse
        analyse(args.output)
    print('Completed:',args.output)

if __name__=='__main__':main()
