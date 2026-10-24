"""Validate fast/native inference and cached prediction before benchmarking.

Place in scripts/. Example (after the predictor adapter exists):
 python scripts/validate_inference.py --model models/xgb.joblib \
   --module xgb_model.predictor --rows data/processed/ha1/validation.csv \
   --heuristic ha1 --threshold 0.43 --output results/xgb_inference.json
Threshold above is illustrative: pass the actual validation-selected threshold.
Use neural_network.predictor for the NN. Dependencies: numpy and root config.py.

Adapter contract extends benchmark.py's predictor interface:
 load_predictor(path, *, cache_enabled, validate_fast=False) -> fresh predictor
 predictor.feature_names, predictor.heuristic
 predictor.predict_fit_probability(X) -> fast P(fit=1), shape (n,)
 predictor.predict_native_fit_probability(X) -> native P(fit=1), shape (n,)
Native must call the ORIGINAL library model (including the SAME fitted NN
scaler), not alias the fast implementation; disable dropout/training mode.
It must bypass caches. The adapter must not call this CLI recursively.

Checks: uncached fast, cached cold/warm, reversed row order, individual rows,
and overwriting one reused input buffer with different real feature rows.
Every output must be finite, in [0,1], input-preserving and close to native.
Default tolerance: abs error <= 1e-6 + 1e-5*abs(native probability).
ALL threshold decisions (p >= threshold) must agree, even within tolerance.
This validates inference equivalence, not classifier accuracy or packing.
Passing is evidence only for the supplied rows, thresholds and tested shapes.

Reads a deterministic reservoir sample from CSV (default 10,000 rows; 0=all).
Uses identical seed and input CSV for both models to get the same sample.
Reports sample indices/hash, model/CSV hashes, per-check errors and decisions.
A FAILED numerical check saves a report and exits 1. Interface/input errors
exit 2. Existing reports are never overwritten. No timings are reported.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
from config import CONFIG, get_feature_names


def _probabilities(p,n):
    p=np.asarray(p)
    if p.shape!=(n,) or p.dtype.kind not in 'fiu':
        raise ValueError('Predictor must return numeric probabilities with shape (n,).')
    p=p.astype(float,copy=False)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid probability values.')
    return p.copy()


def _call(function,X):
    before=X.copy()
    p=_probabilities(function(X),len(X))
    if not np.array_equal(X,before):
        raise ValueError('Predictor modified its input matrix.')
    return p


def validate_inference(factory,X,*,heuristic=CONFIG.heuristic,thresholds,
                       atol=1e-6,rtol=1e-5,single_rows=100):
    """Factory accepts cache_enabled=bool and returns a new adapter instance."""
    names=get_feature_names(heuristic)
    X=np.asarray(X,dtype=np.float64).copy()
    if X.ndim!=2 or X.shape[1]!=len(names) or not len(X) or not np.isfinite(X).all():
        raise ValueError('Supply nonempty, finite raw features with the configured schema.')
    thresholds=list(thresholds)
    if not thresholds or any(not np.isfinite(t) or not 0<=t<=1 for t in thresholds):
        raise ValueError('Supply thresholds in [0,1].')
    if any(not np.isfinite(v) or v<0 for v in (atol,rtol)):
        raise ValueError('Tolerances must be finite and nonnegative.')
    if type(single_rows) is not int or single_rows<1:
        raise ValueError('single_rows must be a positive integer.')
    def load(cached):
        obj=factory(cache_enabled=cached)
        if tuple(getattr(obj,'feature_names',()))!=names or getattr(obj,'heuristic',None)!=heuristic:
            raise ValueError('Adapter feature schema/heuristic mismatch.')
        for method in ('predict_fit_probability','predict_native_fit_probability'):
            if not callable(getattr(obj,method,None)):
                raise ValueError(f'Adapter needs {method}(X).')
        return obj
    uncached=load(False)
    native=_call(uncached.predict_native_fit_probability,X)
    checks=[]
    def compare(name,actual,expected):
        errors=np.abs(actual-expected)
        decisions=[dict(threshold=float(t),mismatches=int(np.sum((actual>=t)!=(expected>=t)))) for t in thresholds]
        bad=int(np.sum(errors>atol+rtol*np.abs(expected)))
        checks.append(dict(check=name,rows=len(actual),max_abs_error=float(errors.max()),
                           mean_abs_error=float(errors.mean()),outside_tolerance=bad,
                           decisions=decisions,passed=bad==0 and all(d['mismatches']==0 for d in decisions)))
    compare('fast_uncached_batch',_call(uncached.predict_fit_probability,X),native)
    cached=load(True)
    if cached is uncached:raise ValueError('Factory must return fresh instances.')
    compare('cached_cold_batch',_call(cached.predict_fit_probability,X),native)
    compare('cached_warm_batch',_call(cached.predict_fit_probability,X),native)
    compare('cached_reversed_rows',_call(cached.predict_fit_probability,X[::-1].copy()),native[::-1])
    k=min(single_rows,len(X))
    indices=np.linspace(0,len(X)-1,k,dtype=int)
    buffer=np.empty((1,X.shape[1]),dtype=float)
    results=[]
    # Include one row again at the end to exercise returning to a previous key.
    indices=np.concatenate([indices,indices[:1]])
    for index in indices:
        buffer[0]=X[index]
        results.append(_call(cached.predict_fit_probability,buffer)[0])
    compare('cached_single_reused_buffer',np.asarray(results),native[indices])
    # Native batching is also checked against individual native calls.
    results=[_call(uncached.predict_native_fit_probability,X[i:i+1].copy())[0] for i in indices]
    compare('native_single_vs_batch',np.asarray(results),native[indices])
    return dict(passed=all(c['passed'] for c in checks),heuristic=heuristic,
                feature_names=list(names),rows=len(X),atol=atol,rtol=rtol,
                thresholds=[float(t) for t in thresholds],checks=checks,
                sampled_features_sha256=hashlib.sha256(X.astype('<f8').tobytes()).hexdigest())


def load_feature_rows(path,heuristic,max_rows=10000,seed=42):
    """Reservoir sample without retaining a potentially huge CSV in memory."""
    if type(max_rows) is not int or max_rows<0:raise ValueError('max_rows must be >=0.')
    names=get_feature_names(heuristic)
    rng=random.Random(seed);sample=[];total=0
    with Path(path).open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        fields=reader.fieldnames or []
        if len(set(fields))!=len(fields) or any(n not in fields for n in names):
            raise ValueError('CSV is missing features or has duplicate columns.')
        for index,row in enumerate(reader):
            total+=1
            if max_rows==0 or len(sample)<max_rows:
                slot=len(sample)
            else:
                slot=rng.randrange(total)
                if slot>=max_rows:continue
            values=[float(row[name]) for name in names]
            entry=(index,values)
            if slot==len(sample):sample.append(entry)
            else:sample[slot]=entry
    if not sample:raise ValueError('CSV contains no feature rows.')
    sample.sort(key=lambda pair:pair[0])
    return np.asarray([r[1] for r in sample]),[r[0] for r in sample],total


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--module',required=True)
    parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--heuristic',choices=('ha1','ha2','ha3'),default=CONFIG.heuristic)
    parser.add_argument('--threshold',type=float,nargs='+',required=True)
    parser.add_argument('--max-rows',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=CONFIG.data.random_seed)
    parser.add_argument('--atol',type=float,default=1e-6)
    parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--single-rows',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output report exists; choose a new path.')
    try:
        X,indices,total=load_feature_rows(args.rows,args.heuristic,args.max_rows,args.seed)
        loader=importlib.import_module(args.module).load_predictor
        def factory(*,cache_enabled):
            return loader(args.model,cache_enabled=cache_enabled,validate_fast=False)
        report=validate_inference(factory,X,heuristic=args.heuristic,thresholds=args.threshold,
                                  atol=args.atol,rtol=args.rtol,single_rows=args.single_rows)
        report.update(model=str(args.model),model_sha256=_hash(args.model),module=args.module,
                      source_csv=str(args.rows),csv_sha256=_hash(args.rows),source_rows=total,
                      sampled_row_indices=indices,seed=args.seed,numpy_version=np.__version__,python=sys.version)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
    except (ValueError,TypeError,OSError,ImportError,AttributeError) as error:
        parser.error(str(error))
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
