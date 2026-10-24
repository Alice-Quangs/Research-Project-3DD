"""Shared metrics for NN/XGBoost and packing (Python 3.10+).

Requires numpy and scikit-learn. Positive class 1 means feasible candidate.
Classification metrics are UNWEIGHTED on the natural held-out distribution;
training class weights must not be passed into evaluation. Predictions accept
p >= threshold, exactly as filtering.py. Undefined ratios return None (JSON
null); average precision and AUROC require both classes here.

packing_metrics returns bins, cost, volume utilisation (ratio, not percent),
runtime and seconds/item, matching benchmark.py names. Bin count is the count
of opened bin objects, not distinct bin types. aggregate_packing_rows combines
ONE method/configuration/repetition only; do not pool repeated runs.

permutation_importance returns plot-ready records using the drop in average
precision. Reuse identical X, y, row ordering, seed and repeats for both models.
X must contain raw shared features; the NN predictor applies its fitted scaler.
Permutation is global across supplied rows, not within orders. Use validation
rows for feature selection; use test rows only for final locked-model reporting.
For large data, choose and save one shared held-out sample upstream. No implicit
sampling, resampling or class balancing occurs here. Correlated/derived features
can share importance; permutation may create physically inconsistent feature
combinations. Negative importance is retained. This is predictive importance,
not a causal effect or a measure of end-to-end packing performance.

This module does not render plots, train models or read files on import.
"""

import math
from numbers import Real
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from config import CONFIG, get_feature_names


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _probabilities(values, n):
    p = np.asarray(values)
    if p.shape != (n,) or p.dtype.kind not in 'fiu':
        raise ValueError('Probabilities must be a numeric vector of shape (n,).')
    p = p.astype(float, copy=False)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and in [0,1].')
    return p


def _labels(values):
    y = np.asarray(values)
    if y.ndim != 1 or not len(y) or y.dtype.kind not in 'biuf' or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be a nonempty binary vector containing only 0 and 1.')
    return y.astype(int, copy=False)


def _threshold(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Threshold must be finite and in [0,1].')
    return float(value)


def classification_metrics(y_true, probabilities, threshold):
    """Confusion counts, recall/precision, FPR/FNR, F1, AP, AUROC and Brier."""
    y = _labels(y_true)
    p = _probabilities(probabilities, len(y))
    threshold = _threshold(threshold)
    predicted = p >= threshold
    positive = y == 1
    tp = int(np.sum(predicted & positive)); fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive)); fn = int(np.sum(~predicted & positive))
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    both = len(np.unique(y)) == 2
    return dict(n=len(y), positives=int(y.sum()), negatives=int(len(y)-y.sum()),
                threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=(tp+tn)/len(y), precision=safe_ratio(tp,tp+fp),
                recall=recall, specificity=specificity,
                fpr=safe_ratio(fp,fp+tn), fnr=safe_ratio(fn,fn+tp),
                f1=safe_ratio(2*tp,2*tp+fp+fn),
                balanced_accuracy=(recall+specificity)/2 if both else None,
                average_precision=float(average_precision_score(y,p)) if both else None,
                auroc=float(roc_auc_score(y,p)) if both else None,
                brier_score=float(np.mean((p-y)**2)))


def select_threshold(y_validation, probabilities, *, partition='validation',
                     target_recall=CONFIG.thresholds.target_recall, thresholds=None):
    """Highest configured-grid threshold meeting validation feasible-bin recall.

    Returns selected threshold, metrics and full sweep. No test/training input
    is permitted by the partition flag; callers must also verify row provenance.
    Requires both classes. Does not estimate weights or retrain the classifier.
    """
    if partition != 'validation':
        raise ValueError('Thresholds must be selected on validation rows only.')
    y = _labels(y_validation)
    p = _probabilities(probabilities, len(y))
    if len(np.unique(y)) != 2:
        raise ValueError('Validation threshold selection requires both classes.')
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0,1].')
    if thresholds is None:
        settings = CONFIG.thresholds
        CONFIG.validate()
        if settings.selection_method != 'highest_threshold_meeting_recall_target':
            raise ValueError('Unsupported threshold selection method.')
        steps = math.floor((settings.grid_max-settings.grid_min)/settings.grid_step + 1e-9)
        thresholds = [settings.grid_min+i*settings.grid_step for i in range(steps+1)]
        thresholds = [min(settings.grid_max,t) for t in thresholds]
        if thresholds[-1] < settings.grid_max-1e-12:
            thresholds.append(settings.grid_max)
    grid = sorted(set(_threshold(t) for t in thresholds))
    if not grid:
        raise ValueError('Threshold grid is empty.')
    # Sort once, avoiding repeated classifier calls or AUROC/AP calculations.
    positives = np.sort(p[y==1]); negatives = np.sort(p[y==0])
    sweep = []
    for t in grid:
        tp = len(positives)-int(np.searchsorted(positives,t,side='left'))
        fp = len(negatives)-int(np.searchsorted(negatives,t,side='left'))
        sweep.append(dict(threshold=t, recall=tp/len(positives),
                          fpr=fp/len(negatives), precision=safe_ratio(tp,tp+fp)))
    eligible = [row for row in sweep if row['recall'] >= target_recall]
    if not eligible:
        raise ValueError('No supplied threshold meets target recall.')
    chosen = eligible[-1]['threshold']
    return dict(threshold=chosen, target_recall=target_recall, partition=partition,
                metrics=classification_metrics(y,p,chosen), sweep=sweep)


def _nonnegative(value, name):
    if (isinstance(value, (bool,np.bool_)) or not isinstance(value,Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def packing_metrics(bins, runtime_s, *, expected_items=None):
    """Compute a completed order's metrics from shared Bin objects.

    Uses cached used_volume, which the heuristic must update at each placement.
    Counts placed_items independently to detect silently dropped items.
    """
    bins = list(bins)
    if len({id(b) for b in bins}) != len(bins):
        raise ValueError('Duplicate bin references would double-count bins and cost.')
    runtime_s = _nonnegative(runtime_s,'runtime_s')
    items = sum(len(b.placed_items) for b in bins)
    if expected_items is not None and items != expected_items:
        raise ValueError('Placed item count differs from expected_items.')
    capacity = sum(_nonnegative(b.volume,'bin volume') for b in bins)
    used = sum(_nonnegative(b.used_volume,'used volume') for b in bins)
    cost = sum(_nonnegative(b.cost,'cost') for b in bins)
    if any(b.volume <= 0 or b.used_volume > b.volume*(1+1e-9) for b in bins):
        raise ValueError('Invalid bin capacity or used volume.')
    return dict(items=items, bins=len(bins), cost=cost, runtime_s=runtime_s,
                item_volume=used, bin_volume=capacity,
                utilisation=safe_ratio(used,capacity), seconds_per_item=safe_ratio(runtime_s,items))


def aggregate_packing_rows(rows):
    """Aggregate one benchmark run; utilisation uses volumes, not mean ratios."""
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to aggregate.')
    for key in ('method','min_bins','threshold','repetition'):
        if len({row.get(key) for row in rows}) > 1:
            raise ValueError(f'Mixed {key} values; aggregate each run separately.')
    if all('order_id' in row for row in rows) and len({r['order_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate order IDs in one run.')
    keys = ('items','bins','cost','runtime_s','item_volume','bin_volume')
    totals = {key:sum(_nonnegative(row[key],key) for row in rows) for key in keys}
    for key in ('items','bins'):
        if any(float(row[key]) != int(row[key]) for row in rows):
            raise ValueError(f'{key} must be whole counts.')
        totals[key] = int(totals[key])
    return dict(orders=len(rows), **totals,
                utilisation=safe_ratio(totals['item_volume'],totals['bin_volume']),
                seconds_per_item=safe_ratio(totals['runtime_s'],totals['items']))


def compare_to_baseline(result, baseline):
    """Compare matched-order totals (or matched median totals) supplied by caller.

    Positive cost/bin changes mean more resource use; speedup >1 means faster.
    This is baseline comparison, never an exact-solver optimality gap.
    """
    for record in (result,baseline):
        for key in ('cost','bins','runtime_s'):
            _nonnegative(record[key],key)
    comparison = dict(speedup_vs_baseline=safe_ratio(baseline['runtime_s'],result['runtime_s']))
    for key in ('cost','bins'):
        delta = result[key]-baseline[key]
        ratio = safe_ratio(delta,baseline[key])
        comparison[f'{key}_change_vs_baseline'] = delta
        comparison[f'{key}_change_pct_vs_baseline'] = 100*ratio if ratio is not None else None
    return comparison


def permutation_importance(predictor, X, y_true, *, heuristic=CONFIG.heuristic,
                           repeats=5, seed=CONFIG.data.random_seed):
    """Return AP decrease by feature, with population SD and each repeat's value.

    Requires a deterministic predictor; inference caches must key on actual
    feature values, never array identity. Input X is never modified.
    """
    names = get_feature_names(heuristic)
    if tuple(getattr(predictor,'feature_names',())) != names or getattr(predictor,'heuristic',None) != heuristic:
        raise ValueError('Predictor feature schema/heuristic mismatch.')
    y = _labels(y_true)
    X = np.asarray(X,dtype=float)
    if X.shape != (len(y),len(names)) or not np.isfinite(X).all():
        raise ValueError('X must be a finite raw matrix matching labels and feature schema.')
    if len(np.unique(y)) != 2:
        raise ValueError('AP permutation importance requires both classes.')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer.')
    baseline = float(average_precision_score(y,_probabilities(predictor.predict_fit_probability(X),len(y))))
    rng = np.random.default_rng(seed)
    work = X.copy()
    drops = np.empty((len(names),repeats))
    # Same permutation for every feature in a repeat; repeatable across models.
    for repeat in range(repeats):
        permutation = rng.permutation(len(y))
        for index in range(len(names)):
            work[:,index] = X[permutation,index]
            score = average_precision_score(y,_probabilities(predictor.predict_fit_probability(work),len(y)))
            drops[index,repeat] = baseline-float(score)
            work[:,index] = X[:,index]
    records = [dict(feature=name, importance_mean=float(drops[i].mean()),
                    importance_std=float(drops[i].std(ddof=0)),
                    importance_repeats=drops[i].tolist()) for i,name in enumerate(names)]
    return dict(scoring='average_precision', baseline_score=baseline, rows=len(y),
                heuristic=heuristic, seed=seed, repeats=repeats,
                features=sorted(records,key=lambda row:row['importance_mean'],reverse=True))
