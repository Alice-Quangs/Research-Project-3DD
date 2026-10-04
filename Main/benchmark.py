"""Shared packing benchmark; run from the repository root (Python 3.10+).

Examples, once heuristic and predictor modules exist:
  python benchmark.py --methods baseline
  python benchmark.py --methods baseline xgboost neural_network \
    --xgb-model models/xgb.joblib --nn-model models/nn.joblib \
    --xgb-threshold 0.43 --nn-threshold 0.6 --min-bins 20
  python benchmark.py --partition validation --sweep \
    --methods baseline xgboost --xgb-model models/xgb.joblib --xgb-threshold 0.43

Threshold numbers above are examples, NOT recommended/selected values.
Sweep filter-start values on validation orders; use a fixed selected value
on test orders. No threshold selection or model fitting occurs here.

Predictor module contract (model-specific files created separately):
  load_predictor(path, *, cache_enabled, validate_fast) -> predictor
The loader must verify the model's feature schema/heuristic and fast/native
prediction equivalence. Returned predictor must expose feature_names in exact
column order, heuristic ('ha1', 'ha2', 'ha3'), and predict_fit_probability(X).
Each loader call must return a FRESH instance with an EMPTY probability cache.
Default modules: xgb_model.predictor and neural_network.predictor. Avoid naming
an importable local package 'xgboost': it shadows the installed XGBoost library.
Override paths with --xgb-module / --nn-module if needed.

Timing: perf_counter around each complete order's packing loop, including
features, prediction, filtering, geometry, state updates and diagnostic counters.
Exclude imports, model loading, data loading, output and final metric calculation.
Use the same wrapper for baseline and learned methods. Fresh bins per order;
fresh predictor per method/repetition/minimum. Caches may persist across orders
within one run. These are cold-start runs, with no warm-up. Method order rotates
across repetitions to reduce systematic timing-order effects.

Outputs: per_order.csv, per_run.csv, summary.json, metadata.json in a new folder.
Summary runtime is the median of WHOLE-RUN totals, never a sum of per-order
medians. Speedup = baseline median runtime / model median runtime. Cost change
is relative to the unfiltered baseline, not an optimality gap. Utilisation is
sum(item volumes) / sum(opened bin volumes), including all orders.
Candidate counts are shortlist sizes, NOT actual geometry-check counts.
Classification accuracy/recall need labelled rows and are not measured here.
"""

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

from bin_types import bin_types
from config import CONFIG, get_feature_names
from data_loader import load_orders
from filtering import pack_item_filtered
from splits import ensure_split_manifest, select_orders, dataset_fingerprint


@dataclass(frozen=True)
class Method:
    name: str
    predictor_factory: object = None
    threshold: float | None = None


def _ratio(a, b):
    return a / b if b else None


def benchmark_orders(orders, heuristic_module, methods, *, heuristic=CONFIG.heuristic,
                     min_bins_values=None, repetitions=CONFIG.benchmark.repetitions,
                     partition='test'):
    """Benchmark already-partitioned canonical orders; return rows/runs/summary.

    CLI validates the full saved manifest. API callers must partition first.
    Factories are zero-argument callables returning fresh, validated predictors.
    """
    if partition not in ('validation', 'test'):
        raise ValueError('Benchmark partition must be validation or test.')
    orders = list(orders)
    if not orders or len({o[0] for o in orders}) != len(orders):
        raise ValueError('Provide nonempty orders with unique IDs.')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer.')
    methods = list(methods)
    if not methods or len({m.name for m in methods}) != len(methods):
        raise ValueError('Methods must be nonempty with unique names.')
    if not any(m.name == 'baseline' for m in methods):
        raise ValueError('Include baseline for cost/runtime comparisons.')
    names = get_feature_names(heuristic)
    learned = [m for m in methods if m.name != 'baseline']
    for m in methods:
        if m.name == 'baseline':
            if m.predictor_factory is not None:
                raise ValueError('Baseline must not have a predictor.')
        elif (not callable(m.predictor_factory) or m.threshold is None
              or isinstance(m.threshold, bool) or not math.isfinite(m.threshold)
              or not 0 <= m.threshold <= 1):
            raise ValueError('Learned methods need a factory and threshold in [0,1].')
    minima = list(min_bins_values) if min_bins_values is not None else [CONFIG.filtering.selected_min_bins]
    if learned and (not minima or len(set(minima)) != len(minima)
                    or any(type(v) is not int or v < 1 for v in minima)):
        raise ValueError('Choose positive min_bins values; no winner is selected by default.')
    if learned and partition == 'test' and len(minima) != 1:
        raise ValueError('Run filter-start sweeps on validation, not test orders.')
    jobs = [(m, None) for m in methods if m.name == 'baseline']
    jobs += [(m, minimum) for m in learned for minimum in minima]
    rows, runs = [], []
    for repetition in range(1, repetitions + 1):
        offset = (repetition - 1) % len(jobs)
        for method, minimum in jobs[offset:] + jobs[:offset]:
            predictor = method.predictor_factory() if method.predictor_factory else None
            if method.name != 'baseline':
                if predictor is None or tuple(getattr(predictor, 'feature_names', ())) != names:
                    raise ValueError(f'{method.name}: predictor feature schema mismatch.')
                if getattr(predictor, 'heuristic', None) != heuristic:
                    raise ValueError(f'{method.name}: predictor heuristic mismatch.')
            batch = []
            for order_id, sequence in orders:
                bins = []
                activations = fallbacks = available = candidates = rejected = 0
                start = time.perf_counter()
                for item in sequence:
                    _, decision = pack_item_filtered(
                        heuristic_module, bin_types, bins, item, predictor,
                        threshold=method.threshold, min_bins_to_filter=minimum,
                        heuristic=heuristic)
                    activations += int(decision.filtering_applied)
                    fallbacks += int(decision.fallback_used)
                    available += decision.open_bin_count
                    candidates += len(decision.candidate_indices)
                    if decision.filtering_applied:
                        rejected += decision.open_bin_count - len(decision.accepted_indices)
                elapsed = time.perf_counter() - start
                if sum(len(b.placed_items) for b in bins) != len(sequence):
                    raise RuntimeError(f'{order_id}: placed-item count mismatch.')
                volume = sum(it['length'] * it['width'] * it['height'] for it in sequence)
                capacity = sum(b.volume for b in bins)
                row = dict(method=method.name, min_bins=minimum, threshold=method.threshold,
                           repetition=repetition, order_id=order_id, items=len(sequence),
                           bins=len(bins), cost=sum(b.cost for b in bins), runtime_s=elapsed,
                           item_volume=volume, bin_volume=capacity,
                           utilisation=_ratio(volume, capacity), filter_activations=activations,
                           empty_shortlist_fallbacks=fallbacks, available_candidates=available,
                           shortlisted_candidates=candidates, raw_model_rejections=rejected)
                batch.append(row)
            rows.extend(batch)
            totals = {key: sum(row[key] for row in batch) for key in
                      ('items', 'bins', 'cost', 'runtime_s', 'item_volume', 'bin_volume',
                       'filter_activations', 'empty_shortlist_fallbacks', 'available_candidates',
                       'shortlisted_candidates', 'raw_model_rejections')}
            runs.append(dict(method=method.name, min_bins=minimum, threshold=method.threshold,
                             repetition=repetition, orders=len(orders), **totals,
                             utilisation=_ratio(totals['item_volume'], totals['bin_volume']),
                             seconds_per_item=_ratio(totals['runtime_s'], totals['items'])))
    summaries = []
    for method, minimum in jobs:
        group = [r for r in runs if r['method'] == method.name and r['min_bins'] == minimum]
        summaries.append(dict(method=method.name, min_bins=minimum, threshold=method.threshold,
                              repetitions=repetitions,
                              median_runtime_s=statistics.median(r['runtime_s'] for r in group),
                              min_runtime_s=min(r['runtime_s'] for r in group),
                              max_runtime_s=max(r['runtime_s'] for r in group),
                              median_cost=statistics.median(r['cost'] for r in group),
                              median_bins=statistics.median(r['bins'] for r in group),
                              median_utilisation=statistics.median(r['utilisation'] for r in group)
                                  if all(r['utilisation'] is not None for r in group) else None,
                              packing_totals_consistent=len({(r['cost'], r['bins']) for r in group}) == 1))
    baseline = next(s for s in summaries if s['method'] == 'baseline')
    for summary in summaries:
        summary['speedup_vs_baseline'] = _ratio(baseline['median_runtime_s'], summary['median_runtime_s'])
        delta = _ratio(summary['median_cost'] - baseline['median_cost'], baseline['median_cost'])
        summary['cost_change_pct_vs_baseline'] = 100 * delta if delta is not None else None
    return rows, runs, summaries


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heuristic', choices=('ha1','ha2','ha3'), default=CONFIG.heuristic)
    parser.add_argument('--heuristic-module')
    parser.add_argument('--dataset', type=Path, default=CONFIG.paths.raw_dataset)
    parser.add_argument('--split-manifest', type=Path, default=CONFIG.paths.split_manifest)
    parser.add_argument('--partition', choices=('validation','test'), default='test')
    parser.add_argument('--methods', nargs='+', choices=('baseline','xgboost','neural_network'),
                        default=['baseline','xgboost','neural_network'])
    parser.add_argument('--min-bins', type=int, default=CONFIG.filtering.selected_min_bins)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--repetitions', type=int, default=CONFIG.benchmark.repetitions)
    parser.add_argument('--xgb-model', type=Path)
    parser.add_argument('--nn-model', type=Path)
    parser.add_argument('--xgb-threshold', type=float)
    parser.add_argument('--nn-threshold', type=float)
    parser.add_argument('--xgb-module', default='xgb_model.predictor')
    parser.add_argument('--nn-module', default='neural_network.predictor')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.sweep and (args.partition != 'validation' or args.min_bins is not None):
        parser.error('--sweep requires validation and no selected/explicit --min-bins.')
    if not args.split_manifest.is_file():
        parser.error('Create the split manifest with splits.py first.')
    orders = load_orders(args.dataset)
    manifest = ensure_split_manifest(orders, args.split_manifest)
    orders = select_orders(orders, manifest, args.partition)
    heuristic_module = importlib.import_module(args.heuristic_module or f'heuristics.{args.heuristic}')
    methods, artifacts = [], {}
    for name in args.methods:
        if name == 'baseline':
            methods.append(Method(name))
            continue
        path, threshold, module_name = ((args.xgb_model,args.xgb_threshold,args.xgb_module)
                                       if name == 'xgboost' else
                                       (args.nn_model,args.nn_threshold,args.nn_module))
        if path is None or threshold is None:
            parser.error(f'{name} requires a model path and validation-selected threshold.')
        loader = importlib.import_module(module_name).load_predictor
        def factory(loader=loader, path=path):
            return loader(path, cache_enabled=CONFIG.benchmark.probability_cache_enabled,
                          validate_fast=CONFIG.benchmark.validate_fast_predictions)
        methods.append(Method(name, factory, threshold))
        artifacts[name] = dict(path=str(path), sha256=_file_hash(path), module=module_name)
    minimums = CONFIG.filtering.min_bins_candidates if args.sweep else [args.min_bins]
    output = args.output or CONFIG.paths.results_dir / args.heuristic / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists():
        parser.error('Output directory already exists; choose a new directory.')
    rows, runs, summary = benchmark_orders(orders, heuristic_module, methods,
        heuristic=args.heuristic, min_bins_values=minimums,
        repetitions=args.repetitions, partition=args.partition)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output/'per_order.csv', rows)
    _write_csv(output/'per_run.csv', runs)
    metadata = dict(heuristic=args.heuristic, heuristic_module=heuristic_module.__name__,
                    partition=args.partition, order_ids=[o[0] for o in orders],
                    partition_dataset_sha256=dataset_fingerprint(orders),
                    split_manifest=str(args.split_manifest), split_sha256=_file_hash(args.split_manifest),
                    models=artifacts, min_bins_values=list(minimums), repetitions=args.repetitions,
                    bin_catalogue=bin_types, config=replace(CONFIG, heuristic=args.heuristic).to_dict(),
                    python=sys.version, platform=platform.platform(),
                    timing='cold-start per-order packing loop; loading and reporting excluded')
    for name, content in [('summary',summary), ('metadata',metadata)]:
        (output/f'{name}.json').write_text(json.dumps(content,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Saved benchmark results to {output}')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
