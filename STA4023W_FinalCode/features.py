"""Shared pre-placement features for NN and XGBoost bin filtering.

Dependencies: numpy, config.py and the shared bin_types.Bin state contract.
Use this module for BOTH training rows and runtime candidate screening.
Columns are defined exclusively by config.get_feature_names: 12 common
features plus seven EMS (ha1/ha2) or EP (ha3) features.

Dimensions/coordinates are mm and volumes are mm^3. Utilisations are ratios
(0 to 1 for current utilisation), not percentages. Projected utilisation may
exceed 1: do not clip it. No placement, feasibility check, scaling, label
generation or bin-state mutation happens here.

The heuristic must update its space representation and call
Bin.register_placement after each actual placement. If it changes spaces
without a placement, call refresh_ems_stats/refresh_ep_stats explicitly.
Feature extraction reads the cached summaries instead of rescanning spaces.

Example:
    X = build_feature_matrix(open_bins, item, heuristic="ha1")
    probabilities = predictor.predict_fit_probability(X)
"""

from collections.abc import Mapping
from numbers import Real
import math

import numpy as np

from config import CONFIG, COMMON_FEATURES, get_feature_names


def _item_dimensions(item):
    """Accept the canonical item dictionary or a (length, width, height) triple."""
    if isinstance(item, Mapping):
        try:
            dimensions = (item["length"], item["width"], item["height"])
        except KeyError as error:
            raise ValueError("Item needs length, width and height keys.") from error
    else:
        try:
            dimensions = tuple(item)
        except TypeError as error:
            raise ValueError("Item must be a dictionary or dimension triple.") from error
        if len(dimensions) != 3:
            raise ValueError("Dimension triples must contain exactly three values.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not math.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError("Item dimensions must be finite positive numbers.")
    return tuple(float(value) for value in dimensions)


def _feature_values(bin_obj, dimensions, item_volume, aspect_ratio, space_names):
    """Build values in schema order; item-only calculations are reused in batches."""
    l, w, h = dimensions
    bin_volume = float(bin_obj.volume)
    used_volume = float(bin_obj.used_volume)
    if not math.isfinite(bin_volume) or bin_volume <= 0:
        raise ValueError("Bin volume must be finite and positive.")
    if not math.isfinite(used_volume) or used_volume < 0:
        raise ValueError("Used bin volume must be finite and nonnegative.")
    common = (
        l, w, h,
        item_volume,
        aspect_ratio,
        float(bin_obj.length),
        float(bin_obj.width),
        float(bin_obj.height),
        float(bin_obj.residual_volume),
        used_volume / bin_volume,
        (used_volume + item_volume) / bin_volume,
        item_volume / bin_volume,
    )
    # EMS dimensions describe the largest-VOLUME EMS, with first-entry
    # tie-breaking, matching Bin.refresh_ems_stats. EP summaries use (x,y,z)
    # points, reordered here according to the z,y,x schema in config.py.
    spaces = tuple(float(getattr(bin_obj, name)) for name in space_names)
    return common + spaces


def _prepare(item, heuristic):
    names = get_feature_names(heuristic)
    dimensions = _item_dimensions(item)
    item_volume = math.prod(dimensions)
    if not math.isfinite(item_volume):
        raise ValueError("Item volume exceeds the supported numeric range.")
    aspect_ratio = max(dimensions) / min(dimensions)
    if not math.isfinite(aspect_ratio):
        raise ValueError("Item aspect ratio exceeds the supported numeric range.")
    return names, dimensions, item_volume, aspect_ratio


def extract_feature_dict(bin_obj, item, heuristic=CONFIG.heuristic):
    """Return an ordered feature dictionary for one candidate bin/item pair.

    This format is useful for logging rows with order IDs and target labels.
    Item metadata, bin costs and target labels are not included as features.
    """
    names, dims, volume, aspect = _prepare(item, heuristic)
    values = _feature_values(bin_obj, dims, volume, aspect,
                             names[len(COMMON_FEATURES):])
    return dict(zip(names, values))


def extract_features(bin_obj, item, heuristic=CONFIG.heuristic):
    """Return a float64 vector of shape (19,) in the configured column order."""
    names, dims, volume, aspect = _prepare(item, heuristic)
    return np.asarray(_feature_values(bin_obj, dims, volume, aspect,
                                     names[len(COMMON_FEATURES):]), dtype=np.float64)


def build_feature_matrix(bins, item, heuristic=CONFIG.heuristic):
    """Return float64 shape (number_of_bins, 19), preserving candidate order.

    Compute item-derived values once per arrival. Empty input returns (0,19).
    Both classifiers receive this same raw matrix; the NN's fitted scaler
    belongs in its own training/prediction pipeline.
    """
    names, dims, volume, aspect = _prepare(item, heuristic)
    bins = list(bins)
    matrix = np.empty((len(bins), len(names)), dtype=np.float64)
    space_names = names[len(COMMON_FEATURES):]
    for index, bin_obj in enumerate(bins):
        matrix[index] = _feature_values(bin_obj, dims, volume, aspect, space_names)
    return matrix
