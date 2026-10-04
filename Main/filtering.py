"""Shared runtime candidate filtering for NN and XGBoost.

Both predictor adapters implement predict_fit_probability(X) -> shape (n,).
Return P(fit=1), not class labels or a two-column predict_proba matrix.
The NN adapter owns its fitted scaler; both receive the same raw features.

Rules from config.py:
* Activate at len(open_bins) >= min_bins_to_filter.
* Accept probability >= threshold, preserving original candidate order.
* Restore ALL open bins only when the predicted shortlist is empty.
* If a nonempty shortlist fails geometry checks, open a new bin without
  retrying rejected bins. Geometry/rotations/new-bin selection stay in the
  heuristic, not the classifier.

No threshold is selected here: supply each model's validation-selected value.
The filter-start sweep is 1, 10, 20, 30, 40. With no explicit value, use
CONFIG.filtering.selected_min_bins; fail if it is still unset.

Example inside a packing loop:
    selected, decision = pack_item_filtered(
        heuristic_module, bin_types, open_bins, item, predictor,
        threshold=validated_threshold, min_bins_to_filter=20, heuristic='ha1')

For baseline packing, pass predictor=None (no threshold/minimum required).
Caching and fast inference belong in predictor adapters, with the same policy
for both models. This module never caches probabilities against mutable bins.
"""

from dataclasses import dataclass
from numbers import Real
import math

import numpy as np

from config import CONFIG, get_feature_names
from features import build_feature_matrix


@dataclass(frozen=True)
class FilterDecision:
    """Diagnostics; indices refer to the ORIGINAL open-bin list.

    accepted_indices are raw model acceptances; candidate_indices include any
    fallback. When inactive, probabilities is None and accepted_indices is
    empty because no predictions were made. No bin objects are copied.
    """
    candidates: tuple
    candidate_indices: tuple
    accepted_indices: tuple
    probabilities: tuple | None
    filtering_applied: bool
    fallback_used: bool
    open_bin_count: int


def _validate_policy():
    settings = CONFIG.filtering
    if (settings.activation_rule != 'open_bin_count_greater_than_or_equal_to_min_bins'
            or settings.fallback_policy != 'restore_all_open_bins_if_shortlist_empty'
            or not settings.preserve_candidate_order
            or settings.retry_rejected_after_nonempty_shortlist_fails):
        raise ValueError('Unsupported filtering policy; update the shared implementation.')


def filter_candidates(open_bins, item, predictor=None, *, threshold=None,
                      min_bins_to_filter=None, heuristic=CONFIG.heuristic):
    """Return candidate references and diagnostics without changing bin state.

    Prediction errors fail explicitly rather than being silently counted as
    fallback. Below activation, neither features nor predictions are computed.
    """
    _validate_policy()
    get_feature_names(heuristic)
    bins = tuple(open_bins)
    indices = tuple(range(len(bins)))
    if predictor is None:
        return FilterDecision(bins, indices, (), None, False, False, len(bins))
    minimum = (CONFIG.filtering.selected_min_bins if min_bins_to_filter is None
               else min_bins_to_filter)
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Set min_bins_to_filter explicitly or select a value in config.py.')
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Provide a finite validation-selected threshold in [0, 1].')
    predict = getattr(predictor, 'predict_fit_probability', None)
    if not callable(predict):
        raise TypeError('Predictor must implement predict_fit_probability(X).')
    if len(bins) < minimum:
        return FilterDecision(bins, indices, (), None, False, False, len(bins))
    matrix = build_feature_matrix(bins, item, heuristic)
    if not np.isfinite(matrix).all():
        raise ValueError('Feature matrix contains non-finite values.')
    raw = np.asarray(predict(matrix))
    if raw.shape != (len(bins),) or raw.dtype.kind not in 'fiu':
        raise ValueError('Predictor must return a numeric probability vector of shape (n_bins,).')
    probabilities = raw.astype(np.float64, copy=False)
    if (not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)):
        raise ValueError('Predicted probabilities must be finite and within [0, 1].')
    accepted = tuple(int(i) for i in np.flatnonzero(probabilities >= threshold))
    fallback = not accepted
    selected = indices if fallback else accepted
    return FilterDecision(tuple(bins[i] for i in selected), selected, accepted,
                          tuple(float(p) for p in probabilities), True, fallback, len(bins))


def pack_item_filtered(heuristic_module, bin_catalogue, open_bins, item,
                       predictor=None, *, threshold=None,
                       min_bins_to_filter=None, heuristic=CONFIG.heuristic):
    """Filter, place once and propagate a newly opened bin to the master list.

    Supports the same placement contract as generate_dataset.py:
      pack_item_unfiltered(bin_catalogue, candidate_bins, item) -> selected_bin
    or legacy pack_item(bin_catalogue, candidate_bins, length, width, height).

    The heuristic must scan candidates in order, update only the selected bin,
    and append exactly one new bin if needed. It must return that bin, or None
    when no catalogue bin can fit. It must maintain caches via
    Bin.register_placement. Candidate bins are references to live master bins;
    the temporary LIST prevents rejected bins from being searched accidentally.
    A failure stops packing rather than silently dropping the item.
    """
    if not isinstance(open_bins, list):
        raise TypeError('open_bins must be a mutable list.')
    modern = getattr(heuristic_module, 'pack_item_unfiltered', None)
    legacy = getattr(heuristic_module, 'pack_item', None)
    if not callable(modern) and not callable(legacy):
        raise TypeError('Heuristic needs pack_item_unfiltered or pack_item.')
    decision = filter_candidates(open_bins, item, predictor, threshold=threshold,
                                 min_bins_to_filter=min_bins_to_filter,
                                 heuristic=heuristic)
    candidates = list(decision.candidates)
    original = tuple(candidates)
    if callable(modern):
        selected = modern(bin_catalogue, candidates, item)
    else:
        selected = legacy(bin_catalogue, candidates,
                          item['length'], item['width'], item['height'])
    if selected is None:
        raise RuntimeError('Item could not be packed; heuristic returned None.')
    if (len(candidates) not in (len(original), len(original) + 1)
            or any(a is not b for a, b in zip(original, candidates))):
        raise RuntimeError('Heuristic changed candidate order or removed existing bins.')
    if len(candidates) == len(original) + 1:
        new_bin = candidates[-1]
        if selected is not new_bin or any(new_bin is b for b in open_bins):
            raise RuntimeError('Invalid newly opened bin returned by heuristic.')
        open_bins.append(new_bin)
    elif not any(selected is b for b in original):
        raise RuntimeError('Selected bin is not a supplied candidate.')
    return selected, decision
