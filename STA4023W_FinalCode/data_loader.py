"""Shared BED-BPP loading for all heuristics and both classifiers.

Canonical output: [(order_id, [item_dict, ...]), ...]. Each item has length,
width, height in mm, weight in kg when available, and optional source metadata.
Numeric item_sequence keys determine arrival order. No random shuffling or
splitting occurs here. Preserve order IDs for the shared split manifest.
"""

import json
import math
from pathlib import Path

from config import CONFIG


def load_dataset(json_path=None):
    """Read the raw order dictionary once, using config.py's path by default."""
    path = Path(json_path) if json_path is not None else CONFIG.paths.raw_dataset
    with path.open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    if not isinstance(dataset, dict):
        raise ValueError(f"Expected an order dictionary in {path}.")
    return dataset


def build_sequence(order, order_id="unknown"):
    """Convert one raw order to the shared list-of-dictionaries item format."""
    sequence = order.get("item_sequence")
    if not isinstance(sequence, dict):
        raise ValueError(f"Order {order_id}: item_sequence must be a dictionary.")
    try:
        keys = sorted(sequence, key=int)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Order {order_id}: sequence keys must be numeric.") from error
    if len({int(key) for key in keys}) != len(keys):
        raise ValueError(f"Order {order_id}: duplicate numeric sequence positions.")
    items = []
    for key in keys:
        source = sequence[key]
        if not isinstance(source, dict):
            raise ValueError(f"Order {order_id}, item {key}: expected a dictionary.")
        item = {}
        for name in ("length", "width", "height"):
            value = source.get(f"{name}/mm")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Order {order_id}, item {key}: invalid {name}/mm.")
            item[name] = value
        weight = source.get("weight/kg")
        if weight is not None and (
            isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0
        ):
            raise ValueError(f"Order {order_id}, item {key}: invalid weight/kg.")
        item["weight"] = weight
        for name in ("product_group", "article", "id"):
            item[name] = source.get(name)
        item["sequence_position"] = int(key)
        items.append(item)
    return items


def load_orders(json_path=None):
    """Load canonical (order_id, sequence) pairs in source order."""
    dataset = load_dataset(json_path)
    return [(order_id, build_sequence(order, order_id))
            for order_id, order in dataset.items()]


def load_sequences_with_ids(json_path=None):
    """Alias for load_orders; items use dictionaries, not positional lists."""
    return load_orders(json_path)


def load_all_sequences(json_path=None):
    """Return item sequences without IDs, using the SAME dictionary format.

    Prefer load_orders for training/evaluation, where order IDs are required.
    This replaces the older loader's [length, width, height, weight] rows.
    """
    return [sequence for _, sequence in load_orders(json_path)]


def load_bed_sequence(json_path=None, order_index=0):
    """Return one order's canonical item sequence (legacy function name)."""
    orders = load_orders(json_path)
    if type(order_index) is not int or not 0 <= order_index < len(orders):
        raise IndexError(f"order_index must be between 0 and {len(orders)-1}.")
    return orders[order_index][1]


def num_orders(json_path=None):
    return len(load_dataset(json_path))
