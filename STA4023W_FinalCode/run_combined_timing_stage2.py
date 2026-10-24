"""Run uninstrumented Stage 2 timing comparisons on the original combined streams.

Place this file and combined_diagnostics_manifest.json in Main and run
python run_combined_timing_stage2.py --check-only
python run_combined_timing_stage2.py
Only the existing project modules perform packing and prediction.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def reconstruct(manifest, source_orders):
    lookup = dict(source_orders)
    orders = []
    seen = set()
    for record in manifest['instances']:
        key = record['order_id']
        if key in seen:
            raise ValueError('Duplicate combined instance ID.')
        seen.add(key)
        sequence = []
        for source in record['source_order_ids']:
            if source not in lookup:
                raise ValueError(f'Missing source order: {source}')
            sequence.extend(dict(item) for item in lookup[source])
        if len(sequence) != record['actual_items']:
            raise ValueError(f'{key}: reconstructed item count differs from original.')
        orders.append((key, sequence, record['target_items']))
    return orders


def orientation_probe(module, heuristic, pack):
    catalogue = [dict(name='orientation_probe', length=100, width=100,
                      height=100, cost=1, max_weight=100)]
    bins = []
    for _ in range(2):
        selected, _ = pack(module, catalogue, bins,
                           dict(length=2, width=3, height=6), heuristic=heuristic)
        box = selected.placed_items[-1]
        dims = tuple(box[d+3]-box[d] for d in range(3))
        if dims != (6, 3, 2):
            raise RuntimeError(f'{heuristic}: Stage 2 orientation probe failed: {dims}.')


def validate_packing(bins, items):
    expected = sorted(tuple(sorted((i['length'], i['width'], i['height']))) for i in items)
    actual = []
    for b in bins:
        boxes = b.placed_items
        for box in boxes:
            dims = tuple(box[d+3]-box[d] for d in range(3))
            actual.append(tuple(sorted(dims)))
            if any(v <= 0 for v in dims):
                raise RuntimeError('Invalid placed dimensions.')
            if any(box[d] < -1e-7 or box[d+3] > cap+1e-7
                   for d, cap in enumerate((b.length, b.width, b.height))):
                raise RuntimeError('Containment violation.')
        for index, a in enumerate(boxes):
            for b2 in boxes[index+1:]:
                if not any(a[d+3] <= b2[d]+1e-7 or b2[d+3] <= a[d]+1e-7 for d in range(3)):
                    raise RuntimeError('Non-overlap violation.')
    if sorted(actual) != expected:
        raise RuntimeError('Packed item dimensions differ from input.')


def run_job(orders, module, filtering, catalogue, predictor, heuristic,
            method, minimum, threshold, repetition):
    rows = []
    for index, (key, sequence, target) in enumerate(orders, 1):
        bins = []
        activations = fallbacks = available = shortlisted = rejected = 0
        start = time.perf_counter()
        for item in sequence:
            _, decision = filtering.pack_item_filtered(
                module, catalogue, bins, item, predictor,
                threshold=threshold, min_bins_to_filter=minimum, heuristic=heuristic)
            activations += int(decision.filtering_applied)
            fallbacks += int(decision.fallback_used)
            available += decision.open_bin_count
            shortlisted += len(decision.candidate_indices)
            if decision.filtering_applied:
                rejected += decision.open_bin_count-len(decision.accepted_indices)
        elapsed = time.perf_counter()-start
        validate_packing(bins, sequence)
        volume = sum(i['length']*i['width']*i['height'] for i in sequence)
        capacity = sum(b.volume for b in bins)
        rows.append(dict(heuristic=heuristic, method=method, min_bins=minimum or 0,
                         threshold=threshold, repetition=repetition, order_id=key,
                         target_items=target, items=len(sequence), bins=len(bins),
                         cost=sum(b.cost for b in bins), runtime_s=elapsed,
                         item_volume=volume, bin_volume=capacity,
                         utilisation=volume/capacity, filter_activations=activations,
                         empty_shortlist_fallbacks=fallbacks, available_candidates=available,
                         shortlisted_candidates=shortlisted, raw_model_rejections=rejected))
        if index == 1 or index % 10 == 0 or index == len(orders):
            print(f'  {heuristic}/{method}/min={minimum or 0}/rep={repetition}: '
                  f'{index}/{len(orders)} streams', flush=True)
    return rows


ADDITIVE = ['items', 'bins', 'cost', 'runtime_s', 'item_volume', 'bin_volume',
            'filter_activations', 'empty_shortlist_fallbacks', 'available_candidates',
            'shortlisted_candidates', 'raw_model_rejections']


def export_results(output, rows):
    write_csv(output/'per_order.csv', rows)
    runs = []
    groups = {}
    for r in rows:
        key = (r['heuristic'], r['method'], r['min_bins'], r['repetition'])
        groups.setdefault(key, []).append(r)
    for (h, m, minimum, rep), group in sorted(groups.items()):
        record = dict(heuristic=h, method=m, min_bins=minimum, repetition=rep, orders=len(group))
        record.update({k:sum(r[k] for r in group) for k in ADDITIVE})
        runs.append(record)
    write_csv(output/'per_run.csv', runs)
    keys = sorted({(r['heuristic'], r['method'], r['min_bins']) for r in rows})
    summaries = []
    for h, m, minimum in keys:
        for target in [0] + sorted({r['target_items'] for r in rows}):
            subset = [r for r in rows if (r['heuristic'],r['method'],r['min_bins']) == (h,m,minimum)
                      and (target == 0 or r['target_items'] == target)]
            if not subset:
                continue
            reps = sorted({r['repetition'] for r in subset})
            record = dict(heuristic=h, method=m, min_bins=minimum,
                          target_items='all' if target==0 else target,
                          repetitions=len(reps), orders_per_repetition=len(subset)//len(reps))
            sums = {k:sum(r[k] for r in subset) for k in ADDITIVE}
            record.update({k:v/len(reps) for k,v in sums.items()})
            record['runtime_s_per_order'] = sums['runtime_s']/len(subset)
            record['activation_pct'] = 100*sums['filter_activations']/sums['items']
            record['candidate_reduction_pct'] = (100*(1-sums['shortlisted_candidates']/sums['available_candidates'])
                                                 if sums['available_candidates'] else 0.0)
            record['fallback_pct_of_activations'] = (100*sums['empty_shortlist_fallbacks']/sums['filter_activations']
                                                    if sums['filter_activations'] else 0.0)
            # Pair with baseline for the SAME repetitions and target group.
            base = [r for r in rows if r['heuristic']==h and r['method']=='baseline'
                    and r['repetition'] in reps and (target==0 or r['target_items']==target)]
            if len(base)==len(subset):
                base_s = sum(r['runtime_s'] for r in base)
                base_c = sum(r['cost'] for r in base)
                record['speedup_vs_baseline'] = base_s/sums['runtime_s']
                record['cost_change_pct'] = 100*(sums['cost']/base_c-1)
            else:
                record.update(speedup_vs_baseline=None, cost_change_pct=None)
            summaries.append(record)
    write_csv(output/'mean_summary.csv', summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('combined_diagnostics_manifest.json'))
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--heuristics', nargs='+', choices=['ha1','ha2','ha3'], default=['ha1','ha2','ha3'])
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--limit-per-target', type=int, help='Smoke test only: first N streams in each target group.')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/combined_timing_stage2'))
    parser.add_argument('--cache-policy', choices=['saved','on','off'], default='saved')
    args = parser.parse_args()
    if args.repetitions < 1 or (args.limit_per_target is not None and args.limit_per_target < 1):
        parser.error('Repetitions and limit must be positive.')
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from data_loader import load_orders
    from splits import dataset_fingerprint
    from bin_types import bin_types
    import filtering
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    dataset = args.dataset or root/'data/raw/bed-bpp_v1.json'
    orders = reconstruct(manifest, load_orders(dataset))
    fingerprint = dataset_fingerprint([(key,seq) for key,seq,_ in orders])
    if fingerprint != manifest['orders_fingerprint']:
        raise RuntimeError('Combined-stream fingerprint differs from original Stage 2 streams.')
    if args.limit_per_target is not None:
        counts = {}
        chosen = []
        for record in orders:
            t = record[2]
            if counts.get(t,0) < args.limit_per_target:
                chosen.append(record)
                counts[t] = counts.get(t,0)+1
        orders = chosen
    modules, models, sources = {}, {}, {}
    for h in args.heuristics:
        expected = manifest['heuristics'][h]
        module = importlib.import_module(f'heuristics.{h}')
        if file_hash(module.__file__) != expected['heuristic_sha256']:
            raise RuntimeError(f'{h}: heuristic differs from earlier combined experiments.')
        orientation_probe(module,h,filtering.pack_item_filtered)
        modules[h] = module
        sources[h] = file_hash(module.__file__)
        for method in ['neural_network','xgboost']:
            folder = 'neural_network' if method=='neural_network' else 'xgb_model'
            loader = importlib.import_module(f'{folder}.predictor').load_predictor
            path = root/'models'/h/folder/'model.joblib'
            previous = expected['models'][method]
            if file_hash(path) != previous['sha256']:
                raise RuntimeError(f'{h}/{method}: saved model differs from earlier combined experiments.')
            cache = expected['cache_enabled'] if args.cache_policy=='saved' else args.cache_policy=='on'
            predictor = loader(path, cache_enabled=cache, validate_fast=True)
            threshold = float(predictor.threshold)
            if predictor.heuristic != h or not math.isclose(threshold,previous['threshold'],abs_tol=1e-12):
                raise RuntimeError(f'{h}/{method}: saved threshold or heuristic mismatch.')
            models[h,method] = (loader,path,threshold,cache)
    print(f'Checks passed: {len(orders)} original combined streams; '
          f'{len(args.heuristics)} heuristics; NN and XGBoost; activation thresholds 1 and 40.', flush=True)
    if args.check_only:
        return
    output = args.output if args.output.is_absolute() else root/args.output
    output.mkdir(parents=True,exist_ok=True)
    source_files = [Path(__file__),root/'filtering.py',root/'features.py',root/'bin_types.py',
                    root/'config.py',root/'data_loader.py',root/'splits.py',
                    root/'xgb_model/predictor.py',root/'neural_network/predictor.py']
    metadata = dict(experiment='combined_timing_stage2',instrumentation_enabled=False,
                    repetitions=args.repetitions,heuristics=args.heuristics,
                    min_bins=[1,40],instance_ids=[r[0] for r in orders],
                    manifest_sha256=file_hash(args.manifest),dataset_sha256=file_hash(dataset),
                    original_combined_fingerprint=fingerprint,heuristic_hashes=sources,
                    source_hashes={str(p.resolve()):file_hash(p) for p in source_files},
                    models={f'{h}/{m}':dict(sha256=file_hash(p),threshold=t,cache_enabled=c)
                            for (h,m),(_,p,t,c) in models.items()},
                    python=sys.version,platform=platform.platform(),
                    timing='Uninstrumented packing loop; model loading, reporting and final geometry validation excluded.',
                    test_streams='Diagnostic sensitivity comparison only; not a new threshold-selection sweep.')
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise RuntimeError('Output settings differ; choose a new output directory.')
    elif any(output.iterdir()):
        raise RuntimeError('Output directory is nonempty without matching metadata.')
    else:
        atomic_json(meta_path, metadata)
    checkpoints = output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    jobs = [(h,m,t) for h in args.heuristics
            for m,t in [('baseline',None),('neural_network',1),('xgboost',1),
                        ('neural_network',40),('xgboost',40)]]
    rows = []
    for rep in range(1,args.repetitions+1):
        offset = (rep-1)%len(jobs)
        for h,m,minimum in jobs[offset:]+jobs[:offset]:
            checkpoint = checkpoints/f'{h}_{m}_{minimum or 0}_rep{rep}.json'
            if checkpoint.exists():
                batch = json.loads(checkpoint.read_text())
                print(f'Resuming: using completed {checkpoint.name}',flush=True)
            else:
                predictor = None
                threshold = None
                if m != 'baseline':
                    loader,path,threshold,cache = models[h,m]
                    predictor = loader(path,cache_enabled=cache,validate_fast=True)
                batch = run_job(orders,modules[h],filtering,bin_types,predictor,h,m,minimum,threshold,rep)
                atomic_json(checkpoint,batch)
            rows.extend(batch)
            export_results(output,rows)
    print(f'Saved results: {output}',flush=True)


if __name__ == '__main__':
    main()
