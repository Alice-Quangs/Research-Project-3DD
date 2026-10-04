"""Generate shared, unweighted candidate-bin rows for NN and XGBoost.

Each order starts with fresh bins. For each item, log ALL existing open bins
before placement, then place the item ONCE with the unfiltered heuristic.
Do not log newly opened bins. Features and labels are shared by both models;
sample/class weights are calculated later, using model-fitting rows only.

Heuristic module contract:
    can_pack(bin_obj, item_dict) -> bool
        Pure check using exactly the constraints/placement options implemented
        by that heuristic. Do not modify the bin while checking.
    pack_item_unfiltered(bin_types, open_bins, item_dict) -> selected_bin
        Place once, track any new bin in open_bins and update cached statistics.

For the supplied legacy HA1, the adapter also supports
pack_item(bin_types, open_bins, length, width, height), and derives labels from
all_orientations/fits_in_ems when can_pack is absent. Those legacy labels mean
EMS dimensional fit only; they do not certify stability, weight or blocking.
HA2 and HA3 must implement can_pack; no approximate EP/EMS label is invented.

Split manifest format (exact order IDs, no overlapping partitions):
    {"train": ["order_1", ...], "validation": [...], "test": [...]}
The CLI requires a saved manifest and never creates a new split implicitly.

Example (after heuristic modules and the split manifest are available):
    python generate_dataset.py --heuristic ha1 --split train

Outputs: a streaming CSV plus a .meta.json summary. CSV columns are tracking
metadata, configured features and fit. Feed only the feature columns to models.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import tempfile

from bin_types import bin_types
from config import CONFIG, FEATURE_SCHEMA_VERSION, get_feature_names
from data_loader import load_orders
from features import extract_feature_dict


def load_split_orders(orders, manifest_path, partition):
    """Validate all partitions and return the requested orders in saved order."""
    if partition not in ("train", "validation", "test"):
        raise ValueError("Partition must be train, validation or test.")
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    lookup = dict(orders)
    if len(lookup) != len(orders):
        raise ValueError("Dataset contains duplicate order IDs.")
    seen = set()
    for name in ("train", "validation", "test"):
        ids = manifest.get(name)
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise ValueError(f"Split manifest needs a list of string IDs for {name}.")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Split manifest contains duplicate or overlapping IDs.")
        missing = set(ids).difference(lookup)
        if missing:
            raise ValueError(f"Split manifest references unknown order IDs: {sorted(missing)[:5]}")
        seen.update(ids)
    if seen != set(lookup):
        raise ValueError("Split manifest must partition every dataset order.")
    return [(order_id, lookup[order_id]) for order_id in manifest[partition]]


def _label_checker(module, heuristic):
    if callable(getattr(module, "can_pack", None)):
        return module.can_pack, "heuristic.can_pack"
    if heuristic == "ha1" and all(callable(getattr(module, name, None))
                                  for name in ("all_orientations", "fits_in_ems")):
        def legacy_ems_fit(bin_obj, item):
            orientations = tuple(module.all_orientations(
                item["length"], item["width"], item["height"]))
            return any(module.fits_in_ems(*orientation, *ems)
                       for ems in bin_obj.ems_list for orientation in orientations)
        return legacy_ems_fit, "legacy_ha1_ems_dimensional_fit"
    raise ValueError(f"{heuristic} must provide can_pack(bin_obj, item_dict).")


def _place(module, open_bins, item):
    if callable(getattr(module, "pack_item_unfiltered", None)):
        return module.pack_item_unfiltered(bin_types, open_bins, item)
    if callable(getattr(module, "pack_item", None)):
        return module.pack_item(bin_types, open_bins,
                                item["length"], item["width"], item["height"])
    raise ValueError("Heuristic must provide pack_item_unfiltered or pack_item.")


def iter_training_rows(orders, heuristic_module, heuristic=CONFIG.heuristic):
    """Yield all-bin rows without fitting, sampling, filtering or shuffling.

    Every candidate for an arrival is inspected before _place is called.
    Bin index is its position in the original open-bin list; item index is
    zero-based in the loaded arrival sequence. Neither is a predictor.
    """
    get_feature_names(heuristic)
    check, _ = _label_checker(heuristic_module, heuristic)
    if not any(callable(getattr(heuristic_module, name, None))
               for name in ("pack_item_unfiltered", "pack_item")):
        raise ValueError("Heuristic has no unfiltered placement function.")
    for order_id, sequence in orders:
        open_bins = []
        for item_index, item in enumerate(sequence):
            for bin_index, bin_obj in enumerate(open_bins):
                features = extract_feature_dict(bin_obj, item, heuristic)
                label = check(bin_obj, item)
                # A placement object/tuple is not a boolean label. Require
                # the can_pack adapter to return a scalar boolean explicitly.
                if not isinstance(label, bool):
                    raise TypeError("can_pack must return a Python bool.")
                yield {
                    "order_id": order_id,
                    "item_index": item_index,
                    "bin_index": bin_index,
                    **features,
                    CONFIG.data.target_column: int(label),
                }
            selected = _place(heuristic_module, open_bins, item)
            if selected is None or not any(selected is b for b in open_bins):
                raise RuntimeError(
                    f"Order {order_id}, item {item_index}: placement failed or "
                    "selected bin was not tracked. Dataset generation stopped.")


def generate_dataset(orders, heuristic_module, output_path,
                     heuristic=CONFIG.heuristic, partition="train"):
    """Stream rows to CSV, preserving an existing CSV if generation fails.

    The caller must supply already-partitioned orders (CLI validates the
    manifest). No rows are retained in memory; counts and metadata are saved.
    """
    if partition not in ("train", "validation", "test"):
        raise ValueError("Partition must be train, validation or test.")
    orders = list(orders)
    ids = [order_id for order_id, _ in orders]
    if any(not isinstance(order_id, str) for order_id in ids):
        raise ValueError("Order IDs must be strings, preserving leading zeros.")
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset contains duplicate order IDs.")
    cfg = replace(CONFIG, heuristic=heuristic)
    cfg.validate()
    if (cfg.data.logging_policy != "all_open_bins_before_placement"
            or cfg.data.include_new_bin_rows):
        raise ValueError("This generator implements all-open-bin logging without new-bin rows.")
    _, label_source = _label_checker(heuristic_module, heuristic)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(cfg.data.metadata_columns) + list(cfg.feature_names) + [cfg.data.target_column]
    count = positives = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in iter_training_rows(orders, heuristic_module, heuristic):
                writer.writerow(row)
                count += 1
                positives += row[cfg.data.target_column]
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    ids = [order_id for order_id, _ in orders]
    summary = {
        "heuristic": heuristic,
        "partition": partition,
        "heuristic_module": getattr(heuristic_module, "__name__", "custom_adapter"),
        "label_source": label_source,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(cfg.feature_names),
        "order_ids": ids,
        "order_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "orders": len(orders),
        "items": sum(len(sequence) for _, sequence in orders),
        "rows": count,
        "positive_rows": positives,
        "negative_rows": count - positives,
        "positive_fraction": positives / count if count else None,
        "class_weights_applied": False,
        "bin_catalogue": bin_types,
        "config": cfg.to_dict(),
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heuristic", choices=("ha1", "ha2", "ha3"), default=CONFIG.heuristic)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--dataset", type=Path, default=CONFIG.paths.raw_dataset)
    parser.add_argument("--split-manifest", type=Path, default=CONFIG.paths.split_manifest)
    parser.add_argument("--heuristic-module", help="Default: heuristics.<heuristic>")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.split_manifest.is_file():
        parser.error("Saved split manifest is missing. Create it with splits.py first.")
    orders = load_split_orders(load_orders(args.dataset), args.split_manifest, args.split)
    name = args.heuristic_module or f"heuristics.{args.heuristic}"
    module = importlib.import_module(name)
    output = args.output or CONFIG.paths.processed_dir / args.heuristic / f"{args.split}.csv"
    summary = generate_dataset(orders, module, output, args.heuristic, args.split)
    print(f"Saved {summary['rows']:,} rows from {summary['orders']:,} orders to {output}")
    print(f"Fit: {summary['positive_rows']:,}; no fit: {summary['negative_rows']:,}")
    print(f"Label source: {summary['label_source']}")


if __name__ == "__main__":
    main()
