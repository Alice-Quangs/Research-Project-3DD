"""Run from Main: python run_clairvoyant_stage2.py. See README_clairvoyant.md."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_packing(bins, items):
    expected = sorted(sorted((it['length'], it['width'], it['height'])) for it in items)
    actual = []
    for b in bins:
        boxes = b.placed_items
        for box in boxes:
            if len(box) != 6:
                raise RuntimeError('Expected placed_items as (x1,y1,z1,x2,y2,z2).')
            dims = [box[d+3]-box[d] for d in range(3)]
            if any(v <= 0 for v in dims):
                raise RuntimeError('Nonpositive placed-item dimension.')
            actual.append(sorted(dims))
            if any(box[d] < -1e-7 or box[d+3] > cap+1e-7
                   for d, cap in enumerate((b.length,b.width,b.height))):
                raise RuntimeError('Placement violates containment.')
        for i, a in enumerate(boxes):
            for b2 in boxes[i+1:]:
                if not any(a[d+3] <= b2[d]+1e-7 or b2[d+3] <= a[d]+1e-7 for d in range(3)):
                    raise RuntimeError('Placement violates non-overlap.')
    if sorted(actual) != expected:
        raise RuntimeError('Packed items do not match the input dimensions.')


def check_orientation(module, h, pack):
    catalogue = [dict(name='orientation_probe',length=100,width=100,height=100,cost=1,max_weight=100)]
    bins = []
    for _ in range(2):
        selected, _ = pack(module, catalogue, bins, dict(length=2,width=3,height=6), heuristic=h)
        box = selected.placed_items[-1]
        orientation = tuple(box[d+3]-box[d] for d in range(3))
        if orientation != (6,3,2):
            raise RuntimeError(f'{h}: orientation probe returned {orientation}, expected (6,3,2). '
                               'Install your final Stage 2 heuristic before running. No heuristic is patched by this runner.')


def export_results(output, records):
    exact_rows, comparison_rows, per_run = [], [], []
    for record in records:
        base = {k:record[k] for k in ('order_id','subset_size','time_limit_seconds','previous_optimal_cost')}
        exact_rows.append({**base, **record['exact']})
        per_run.extend(record['per_run'])
        for result in record['comparisons']:
            comparison_rows.append({**base, **record['exact'], **result})
    write_csv(output/'exact_results.csv', exact_rows)
    write_csv(output/'comparison_detailed.csv', comparison_rows)
    write_csv(output/'per_run.csv', per_run)
    groups = {}
    for row in comparison_rows:
        groups.setdefault((row['subset_size'],row['heuristic'],row['method']),[]).append(row)
    summary = []
    for (n,h,m), group in sorted(groups.items()):
        gaps = [r['gap_to_optimum_pct'] for r in group if r['gap_to_optimum_pct'] is not None]
        summary.append(dict(subset_size=n,heuristic=h,method=m,instances=len(group),
            instances_proven_optimal=len(gaps), mean_cost=statistics.mean(r['cost'] for r in group),
            mean_gap_to_optimum_pct=statistics.mean(gaps) if gaps else None,
            max_gap_to_optimum_pct=max(gaps) if gaps else None,
            mean_packing_runtime_s=statistics.mean(r['mean_runtime_s'] for r in group),
            total_filter_activations=sum(r['filter_activations'] for r in group)))
    write_csv(output/'comparison_summary.csv',summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--instances',type=Path,default=root/'clairvoyant_instances.csv')
    parser.add_argument('--dataset',type=Path,default=root/'data/raw/bed-bpp_v1.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--methods',nargs='+',choices=('baseline','neural_network','xgboost'),
                        default=['baseline','neural_network','xgboost'])
    parser.add_argument('--repetitions',type=int,default=7)
    parser.add_argument('--min-bins',type=int,default=40)
    parser.add_argument('--time-limit',type=int,help='Override per-instance solver limit in seconds.')
    parser.add_argument('--limit',type=int,help='Run only the first N manifest instances for a smoke check.')
    parser.add_argument('--check-only',action='store_true',help='Validate instances, orientations and model loading; do not solve.')
    args = parser.parse_args()
    if 'baseline' not in args.methods or len(set(args.methods)) != len(args.methods):
        parser.error('Include baseline, and do not repeat methods.')
    if min(args.repetitions,args.min_bins) < 1 or any(v is not None and v < 1 for v in (args.limit,args.time_limit)):
        parser.error('Counts and limits must be positive.')
    sys.path.insert(0,str(root))
    from bin_types import bin_types
    from data_loader import load_orders
    from filtering import pack_item_filtered
    from clairvoyant_exact import solve_exact
    modules = {h:importlib.import_module(f'heuristics.{h}') for h in ('ha1','ha2','ha3')}
    for h,module in modules.items():
        check_orientation(module,h,pack_item_filtered)
    orders = dict(load_orders(args.dataset))
    instances = list(csv.DictReader(args.instances.open(encoding='utf-8')))
    if args.limit:
        instances = instances[:args.limit]
    seen = set()
    for inst in instances:
        k,n = inst['order_id'],int(inst['subset_size'])
        if (k,n) in seen or k not in orders or len(orders[k]) < n:
            raise ValueError(f'Invalid instance: {k}, {n}')
        seen.add((k,n))
        dims = [[it['length'],it['width'],it['height']] for it in orders[k][:n]]
        if dims != json.loads(inst['item_dimensions_json']):
            raise ValueError(f'{k}/{n}: dataset prefix differs from the original benchmark.')
    artifacts, loaders = {}, {}
    for h in modules:
        for method in args.methods:
            if method == 'baseline':
                continue
            folder = 'xgb_model' if method == 'xgboost' else 'neural_network'
            path = root/'models'/h/folder/'model.joblib'
            loader = importlib.import_module(f'{folder}.predictor').load_predictor
            predictor = loader(path,cache_enabled=False)
            if predictor.heuristic != h:
                raise ValueError('Saved model heuristic mismatch.')
            loaders[h,method] = (loader,path,float(predictor.threshold))
            artifacts[f'{h}/{method}'] = dict(path=str(path),sha256=file_hash(path),threshold=float(predictor.threshold))
    print(f'Checks passed: {len(instances)} instances; orientation probes and requested models.',flush=True)
    if args.check_only:
        return
    try:
        importlib.import_module('ortools.linear_solver.pywraplp')
    except ImportError as error:
        raise RuntimeError('Install solver: python -m pip install ortools') from error
    source_paths = [Path(__file__),root/'clairvoyant_exact.py',root/'filtering.py',root/'bin_types.py',
                    root/'features.py',root/'config.py',root/'data_loader.py']
    source_paths += [Path(m.__file__) for m in modules.values()]
    source_paths += [root/folder/'predictor.py' for folder in ('xgb_model','neural_network') if (root/folder/'predictor.py').is_file()]
    metadata = dict(dataset_sha256=file_hash(args.dataset),instances_sha256=file_hash(args.instances),
                    instance_keys=sorted([list(k) for k in seen]),bin_catalogue=bin_types,models=artifacts,
                    source_hashes={str(p.relative_to(root)):file_hash(p) for p in source_paths},
                    methods=args.methods,repetitions=args.repetitions,min_bins=args.min_bins,time_limit_override=args.time_limit,
                    python=sys.version,platform=platform.platform(),
                    scope='Orthogonal rotations, containment, non-overlap and bin cost; no weight, support/stability or blocking.',
                    timing='Packing loop only; model loading, geometry validation and exact solve excluded.',
                    instance_selection='Fixed historical prefixes; not necessarily restricted to the ML test partition.')
    output = args.output or root/'results/clairvoyant_stage2'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=True)
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise RuntimeError('Output metadata differs. Use a new output folder; changed runs cannot resume old checkpoints.')
    elif any(output.iterdir()):
        raise RuntimeError('Output folder is nonempty without matching metadata; choose a new folder.')
    else:
        atomic_json(meta_path,metadata)
    checkpoints = output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records = []
    jobs = [(h,m) for h in modules for m in args.methods]
    for index,inst in enumerate(instances,1):
        k,n = inst['order_id'],int(inst['subset_size'])
        checkpoint = checkpoints/f'{k}_{n}.json'
        if checkpoint.exists():
            records.append(json.loads(checkpoint.read_text()))
            export_results(output,records)
            print(f'[{index}/{len(instances)}] {k}/{n}: resumed completed instance',flush=True)
            continue
        sequence = orders[k][:n]
        per_run = []
        for repetition in range(1,args.repetitions+1):
            offset = (repetition-1)%len(jobs)
            for h,method in jobs[offset:]+jobs[:offset]:
                predictor,threshold = None,None
                if method != 'baseline':
                    loader,path,threshold = loaders[h,method]
                    predictor = loader(path,cache_enabled=False)
                bins,activations = [],0
                start = time.perf_counter()
                for item in sequence:
                    _,decision = pack_item_filtered(modules[h],bin_types,bins,item,predictor,
                         threshold=threshold,min_bins_to_filter=args.min_bins,heuristic=h)
                    activations += int(decision.filtering_applied)
                elapsed = time.perf_counter()-start
                validate_packing(bins,sequence)
                per_run.append(dict(order_id=k,subset_size=n,heuristic=h,method=method,repetition=repetition,
                    bins=len(bins),cost=sum(b.cost for b in bins),runtime_s=elapsed,
                    threshold=threshold,min_bins=args.min_bins if predictor else None,filter_activations=activations))
        upper = min(r['cost'] for r in per_run)
        limit = args.time_limit or int(inst['time_limit_seconds'])
        print(f'[{index}/{len(instances)}] Solving {k}/{n}; limit {limit}s, upper bound {upper}',flush=True)
        exact = clean(solve_exact([(it['length'],it['width'],it['height']) for it in sequence],bin_types,upper,limit))
        if exact['solver_status'] in ('INFEASIBLE','ABNORMAL','UNBOUNDED'):
            raise RuntimeError(f'Unexpected exact solver status: {exact["solver_status"]}')
        comparisons = []
        for h,method in jobs:
            group = [r for r in per_run if r['heuristic']==h and r['method']==method]
            if len({(r['cost'],r['bins'],r['filter_activations']) for r in group}) != 1:
                raise RuntimeError('Packing outcomes changed across repetitions.')
            cost = group[0]['cost']
            optimum,bound = exact['incumbent_cost'],exact['best_bound']
            if exact['is_optimal'] and cost < optimum-1e-6:
                raise RuntimeError('Heuristic cost is below the exact optimum; check model consistency.')
            comparisons.append(dict(heuristic=h,method=method,cost=cost,bins=group[0]['bins'],
                mean_runtime_s=statistics.mean(r['runtime_s'] for r in group),
                runtime_std_s=statistics.stdev(r['runtime_s'] for r in group) if len(group)>1 else 0,
                filter_activations=group[0]['filter_activations'],
                gap_to_optimum_pct=100*(cost-optimum)/optimum if exact['is_optimal'] and optimum else None,
                certified_gap_upper_pct=100*(cost-bound)/bound if bound and bound>0 else None))
        record = dict(order_id=k,subset_size=n,time_limit_seconds=limit,
                      previous_optimal_cost=float(inst['previous_optimal_cost']),
                      exact=exact,comparisons=comparisons,per_run=per_run)
        atomic_json(checkpoint,record)
        records.append(record)
        export_results(output,records)
        print(f'  {exact["solver_status"]}: cost={exact["incumbent_cost"]}, bound={exact["best_bound"]}',flush=True)
    print(f'Saved results: {output}',flush=True)


if __name__ == '__main__':
    main()
