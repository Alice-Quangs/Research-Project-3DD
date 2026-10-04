"""Create/reuse one order-level split for both classifiers and all heuristics.

Run from the repository root: python splits.py
Defaults come from config.py (64/16/20, seed 42). Output is
 data/splits/order_splits.json, compatible with generate_dataset.py.

Only order IDs are shuffled; item arrival sequences remain unchanged.
IDs are sorted before shuffling, so source dictionary order has no effect.
Counts use largest-remainder rounding (ties: train, validation, test).
Small datasets that would leave a partition empty are rejected.

Existing manifests are validated and reused, never silently regenerated.
Use a different --output for a new experiment. Commit the manifest to Git.
This creates a NEW experimental split, not the historical row-level split.
No class balancing, feature extraction or model fitting happens here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from config import CONFIG
from data_loader import load_orders

PARTITIONS = ("train", "validation", "test")


def _validated_ids(order_ids):
    ids = list(order_ids)
    if any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Order IDs must be nonempty strings; preserve leading zeros.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate order IDs are not allowed.")
    return ids


def _fractions(settings):
    values = [settings.train_fraction, settings.validation_fraction,
              settings.test_fraction]
    if (any(not math.isfinite(x) or not 0 < x < 1 for x in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-9, rel_tol=0)):
        raise ValueError("Split fractions must be positive and sum to one.")
    if type(settings.random_seed) is not int:
        raise ValueError("random_seed must be an integer.")
    if settings.split_unit != "order" or settings.arrival_order != "dataset_sequence":
        raise ValueError("This splitter requires order splits and dataset arrival order.")
    return values


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(orders):
    """Hash loaded items as well as IDs; preserve item order within each order."""
    orders = list(orders)
    _validated_ids(order_id for order_id, _ in orders)
    digest = hashlib.sha256()
    for order in sorted(orders, key=lambda pair: pair[0]):
        digest.update(bytes.fromhex(_digest(order)))
    return digest.hexdigest()


def create_split_manifest(order_ids, settings=CONFIG.data):
    """Return exact ID lists plus metadata without writing files."""
    ids = sorted(_validated_ids(order_ids))
    fractions = _fractions(settings)
    quotas = [len(ids) * fraction for fraction in fractions]
    counts = [math.floor(q) for q in quotas]
    remainder = len(ids) - sum(counts)
    ranking = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in ranking[:remainder]:
        counts[i] += 1
    if any(count == 0 for count in counts):
        raise ValueError("Too few orders for three nonempty partitions at these fractions.")
    id_hash = _digest(ids)
    random.Random(settings.random_seed).shuffle(ids)
    manifest = {}
    start = 0
    for partition, count in zip(PARTITIONS, counts):
        manifest[partition] = ids[start:start + count]
        start += count
    manifest["metadata"] = {
        "schema_version": 1,
        "split_unit": "order",
        "seed": settings.random_seed,
        "fractions": dict(zip(PARTITIONS, fractions)),
        "counts": dict(zip(PARTITIONS, counts)),
        "algorithm": "sorted_ids_python_random_shuffle_largest_remainder_v1",
        "arrival_order": "dataset_sequence",
        "order_ids_sha256": id_hash,
    }
    validate_split_manifest(manifest, ids)
    return manifest


def validate_split_manifest(manifest, order_ids):
    """Reject overlap, duplicates, missing/unknown IDs and empty partitions."""
    expected = set(_validated_ids(order_ids))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    seen = set()
    for partition in PARTITIONS:
        ids = manifest.get(partition)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{partition} must contain a nonempty list of order IDs.")
        ids = _validated_ids(ids)
        if seen.intersection(ids):
            raise ValueError("Split partitions overlap.")
        seen.update(ids)
    if seen != expected:
        raise ValueError("Manifest IDs do not match the dataset exactly.")
    return manifest


def load_split_manifest(path, order_ids):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_split_manifest(json.load(stream), order_ids)


def select_orders(orders, manifest, partition):
    """Return one partition in saved order, without altering item sequences."""
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    orders = list(orders)
    validate_split_manifest(manifest, [order_id for order_id, _ in orders])
    lookup = dict(orders)
    return [(order_id, lookup[order_id]) for order_id in manifest[partition]]


def ensure_split_manifest(orders, path=None, settings=CONFIG.data):
    """Create once, or reuse after validating IDs, content and split settings.

    Existing files without this script's metadata are rejected rather than
    assuming their origin. Use a new output path to create a fresh experiment.
    """
    orders = list(orders)
    ids = _validated_ids(order_id for order_id, _ in orders)
    fractions = _fractions(settings)
    fingerprint = dataset_fingerprint(orders)
    path = Path(path) if path is not None else CONFIG.paths.split_manifest
    if path.exists():
        manifest = load_split_manifest(path, ids)
        meta = manifest.get("metadata", {})
        if not isinstance(meta, dict):
            raise ValueError("Invalid manifest metadata.")
        required = {
            "schema_version": 1,
            "seed": settings.random_seed,
            "fractions": dict(zip(PARTITIONS, fractions)),
            "dataset_sha256": fingerprint,
            "order_ids_sha256": _digest(sorted(ids)),
            "split_unit": "order",
            "arrival_order": "dataset_sequence",
            "counts": {key: len(manifest[key]) for key in PARTITIONS},
        }
        if any(meta.get(key) != value for key, value in required.items()):
            raise ValueError("Existing manifest differs from the dataset/settings or lacks "
                             "metadata. Use a new --output for a new experiment.")
        return manifest
    manifest = create_split_manifest(ids, settings)
    manifest["metadata"]["dataset_sha256"] = fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of an existing split.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CONFIG.paths.raw_dataset)
    parser.add_argument("--output", type=Path, default=CONFIG.paths.split_manifest)
    args = parser.parse_args()
    try:
        orders = load_orders(args.dataset)
        existed = args.output.exists()
        manifest = ensure_split_manifest(orders, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"{'Reused' if existed else 'Created'} split manifest: {args.output}")
    for partition in PARTITIONS:
        print(f"{partition}: {len(manifest[partition]):,} orders")
    print("Reuse this manifest for NN, XGBoost and all heuristics.")


if __name__ == "__main__":
    main()
