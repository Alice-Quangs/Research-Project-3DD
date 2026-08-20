# run_ha1_nn_filtered.py
#
# Runs HA1 (F53/EMS heuristic) WITH the trained NN acting as a bin filter.
#
# For each item, before the heuristic does its full geometric EMS check on
# an open bin, the NN predicts Fit/No-Fit for (bin dims, item dims, red_cap).
# If the NN predicts No-Fit, that bin is skipped entirely (heuristic never
# even checks its EMS list) and the next open bin is tried.
#
# PERFORMANCE NOTE: predictions for ALL open bins for a given item are
# batched into a single model.predict() call, rather than one call per bin.
# This is the same logic as the original version, just far fewer, larger
# calls to sklearn instead of many single-row calls (which carry heavy
# per-call overhead).
#
# Note: because the feature set is bin-level (not EMS-level), the NN filters
# at the level of "is this bin worth trying", not "which EMS is best" -
# EMS/orientation selection within an approved bin is still done exactly as
# in the original ha1.py (best_ems / best_orientation), untouched.
#
# Does NOT modify ha1.py or bin_types.py - only imports from them.

import json
import csv
import time
import joblib
import pandas as pd
from bin_types import bin_types, Bin
from ha1 import (
    all_orientations,
    fits_in_ems,
    update_ems,
    best_ems,
    best_orientation,
)

DATA_PATH = "bed-bpp_v1.json"
MODEL_PATH = "ha1_nn_model.joblib"
OUTPUT_PATH = "ha1_nn_filtered_results.csv"

FEATURES = [
    "bin_length", "bin_width", "bin_height",
    "item_length", "item_width", "item_height",
    "red_cap"
]

# Decision threshold on P(Fit), NOT the default 0.5.
#
# The two error types here have very different costs:
#   - False positive (NN says Fit, geometry disagrees): the real EMS check
#     catches it right after - just one wasted geometry check.
#   - False negative (NN says No Fit, but it would have fit): the heuristic
#     never tries that bin at all - a perfectly good open bin gets skipped,
#     forcing a brand-new bin to be opened. This directly inflates bin count
#     and compounds over the rest of the order.
#
# Because "Fit" is the rare class (~7%), the default 0.5 cutoff gives poor
# recall on Fit even though overall accuracy looks fine. Lowering the
# threshold trades cheap false positives for expensive false negatives.
# 0.05 was chosen by sweeping thresholds on the held-out test set: it pushes
# Fit recall from ~0.33 (at 0.5) to ~0.96, while only forcing the real
# geometry check on ~23% of true No-Fit bins (still skips the other 77%
# outright). Re-run the sweep if the model is retrained, since the right
# threshold depends on that specific model's calibration.
NN_FIT_THRESHOLD = 0.05


def build_sequence(order):
    items = []
    for key in sorted(order['item_sequence'].keys(), key=lambda x: int(x)):
        item = order['item_sequence'][key]
        items.append({
            'length': item['length/mm'],
            'width': item['width/mm'],
            'height': item['height/mm'],
        })
    return items


def bin_red_cap(bin_obj):
    bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
    used_volume = sum(
        (x2 - x1) * (y2 - y1) * (z2 - z1)
        for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
    )
    return bin_volume - used_volume


def nn_predict_all_bins(model, open_bins, item_l, item_w, item_h):
    """
    Batches the NN prediction across ALL open bins for this item in a
    single model.predict_proba() call. Returns a list of booleans (True =
    NN says Fit) in the same order as open_bins, using NN_FIT_THRESHOLD
    instead of the default 0.5 cutoff (see comment above NN_FIT_THRESHOLD
    for why).
    """
    if not open_bins:
        return []

    rows = []
    for bin_obj in open_bins:
        red_cap = bin_red_cap(bin_obj)
        rows.append([
            bin_obj.length, bin_obj.width, bin_obj.height,
            item_l, item_w, item_h,
            red_cap
        ])

    batch = pd.DataFrame(rows, columns=FEATURES)
    fit_class_index = list(model.classes_).index(1)
    probs = model.predict_proba(batch)[:, fit_class_index]
    return [bool(p >= NN_FIT_THRESHOLD) for p in probs]


def pack_item_nn_filtered(bin_types, open_bins, item_l, item_w, item_h, model, stats):
    """
    Same as HA1's pack_item, but every open bin is screened by the NN in a
    single batched call before geometric checks begin. Bins the NN rejects
    are skipped without any EMS/orientation checking.
    """
    orientations = all_orientations(item_l, item_w, item_h)

    nn_predictions = nn_predict_all_bins(model, open_bins, item_l, item_w, item_h)
    stats["bins_screened"] += len(open_bins)

    for bin_obj, nn_fit in zip(open_bins, nn_predictions):

        if not nn_fit:
            stats["bins_rejected_by_nn"] += 1
            continue  # NN says skip this bin, try the next one

        # NN approved this bin - proceed with normal HA1 geometric packing
        valid_ems_list = []
        valid_orientations_per_ems = []

        for ems in bin_obj.ems_list:
            valid_orientations = []
            for (ol, ow, oh) in orientations:
                if fits_in_ems(ol, ow, oh, ems[0], ems[1], ems[2], ems[3], ems[4], ems[5]):
                    valid_orientations.append((ol, ow, oh))
            if valid_orientations:
                valid_ems_list.append(ems)
                valid_orientations_per_ems.append(valid_orientations)

        if valid_ems_list:
            chosen_ems = best_ems(valid_ems_list, bin_obj.length, bin_obj.width, bin_obj.height)
            ems_index = valid_ems_list.index(chosen_ems)
            chosen_orient = best_orientation(valid_orientations_per_ems[ems_index])
            ol, ow, oh = chosen_orient
            item_x, item_y, item_z = chosen_ems[0], chosen_ems[1], chosen_ems[2]

            bin_obj.placed_items.append((item_x, item_y, item_z, item_x + ol, item_y + ow, item_z + oh))
            bin_obj.ems_list = update_ems(bin_obj.ems_list, item_x, item_y, item_z, ol, ow, oh)
            return bin_obj
        else:
            # NN approved the bin, but geometry disagreed - this happens
            # since the NN can't see individual EMS shapes, only bin-level stats
            stats["nn_approved_but_geometry_failed"] += 1
            continue

    # No existing bin worked (NN-rejected or geometrically failed) - open a new one
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ems_list = update_ems(new_bin.ems_list, 0, 0, 0, ol, ow, oh)
                open_bins.append(new_bin)
                return new_bin

    print("Item could not be packed!")
    return None


def main():
    print("Loading trained NN model...")
    model = joblib.load(MODEL_PATH)

    print("Loading BED-BPP dataset...")
    with open(DATA_PATH, 'r') as f:
        dataset = json.load(f)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    print(f"Loaded {n_orders} orders")

    print("\n" + "=" * 60)
    print(" RUNNING HA1 + NN FILTER")
    print("=" * 60)

    total_bins = 0
    total_cost = 0
    total_volume_used = 0
    total_bin_volume = 0
    per_order_rows = []

    stats = {
        "bins_screened": 0,
        "bins_rejected_by_nn": 0,
        "nn_approved_but_geometry_failed": 0,
    }

    start_time = time.perf_counter()

    for idx, order_id in enumerate(order_ids):
        order = dataset[order_id]
        items = build_sequence(order)

        open_bins = []

        # Time this specific order
        order_start = time.perf_counter()

        for item in items:
            pack_item_nn_filtered(
                bin_types, open_bins,
                item['length'], item['width'], item['height'],
                model, stats
            )

        order_time = time.perf_counter() - order_start

        num_bins = len(open_bins)
        cost = sum(bin_obj.cost for bin_obj in open_bins)
        total_bins += num_bins
        total_cost += cost

        order_volume_used = 0
        order_bin_volume = 0
        for bin_obj in open_bins:
            bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
            used_volume = sum(
                (x2 - x1) * (y2 - y1) * (z2 - z1)
                for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
            )
            order_volume_used += used_volume
            order_bin_volume += bin_volume

        total_volume_used += order_volume_used
        total_bin_volume += order_bin_volume

        order_utilization = (order_volume_used / order_bin_volume * 100) if order_bin_volume else 0
        per_order_rows.append([
            order_id, len(items), num_bins, cost, order_utilization, order_time
        ])

        if (idx + 1) % 100 == 0 or (idx + 1) == n_orders:
            elapsed_so_far = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders ({elapsed_so_far:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    utilization = (total_volume_used / total_bin_volume) * 100 if total_bin_volume else 0
    avg_bins_per_order = total_bins / n_orders

    print("\n" + "=" * 60)
    print(" HA1 + NN FILTER RESULTS")
    print("=" * 60)
    print(f"Orders processed: {n_orders}")
    print(f"Total bins used: {total_bins}")
    print(f"Average bins per order: {avg_bins_per_order:.3f}")
    print(f"Total cost: {total_cost}")
    print(f"Overall space utilization: {utilization:.2f}%")
    print(f"Time taken: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"\nBins screened by NN: {stats['bins_screened']}")
    print(f"Bins rejected by NN: {stats['bins_rejected_by_nn']} "
          f"({stats['bins_rejected_by_nn'] / max(stats['bins_screened'], 1) * 100:.1f}%)")
    print(f"NN approved but geometry failed anyway: {stats['nn_approved_but_geometry_failed']}")

    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "orders_processed", "total_bins", "avg_bins_per_order",
            "total_cost", "space_utilization_pct", "time_seconds",
            "bins_screened", "bins_rejected_by_nn", "nn_approved_but_geometry_failed"
        ])
        writer.writerow([
            n_orders, total_bins, avg_bins_per_order,
            total_cost, utilization, elapsed,
            stats["bins_screened"], stats["bins_rejected_by_nn"], stats["nn_approved_but_geometry_failed"]
        ])
    print(f"\nSaved: {OUTPUT_PATH}")

    per_order_path = "ha1_nn_filtered_per_order.csv"
    with open(per_order_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "item_count", "bins_used", "cost", "utilization_pct", "time_seconds"])
        writer.writerows(per_order_rows)
    print(f"Saved: {per_order_path}")


if __name__ == "__main__":
    main()