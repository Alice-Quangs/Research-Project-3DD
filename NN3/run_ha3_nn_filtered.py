# run_ha3_nn_filtered.py
#
# Runs HA3 (Extreme Points / EP heuristic) WITH the trained NN acting as a bin filter.
#
# For each item, before the heuristic does its full geometric EP check on
# an open bin, the NN predicts Fit/No-Fit for (bin dims, item dims, red_cap).
# If the NN predicts No-Fit, that bin is skipped entirely (heuristic never
# even checks its EP list) and the next open bin is tried.
#
# PERFORMANCE NOTE: predictions for ALL open bins for a given item are
# batched into a single model.predict() call, rather than one call per bin.
# This is the same logic as the original version, just far fewer, larger
# calls to sklearn instead of many single-row calls (which carry heavy
# per-call overhead).
#
# Note: because the feature set is bin-level (not EP-level), the NN filters
# at the level of "is this bin worth trying", not "which EP is best" -
# EP/orientation selection within an approved bin is still done exactly as
# in the original ha3.py (first valid EP, best_orientation), untouched.
#
# Does NOT modify ha3.py or bin_types.py - only imports from them.

import json
import csv
import time
import joblib
import pandas as pd
from bin_types import bin_types
from ha3 import (
    Bin,
    all_orientations,
    can_place,
    update_ep,
)

DATA_PATH = "bed-bpp_v1.json"
MODEL_PATH = "ha3_nn_model.joblib"
OUTPUT_PATH = "ha3_nn_filtered_results.csv"

FEATURES = [
    "bin_length", "bin_width", "bin_height",
    "item_length", "item_width", "item_height",
    "red_cap"
]


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
    single model.predict() call. Returns a list of booleans (True = NN
    says Fit) in the same order as open_bins.
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
    predictions = model.predict(batch)
    return [bool(p) for p in predictions]


def pack_item_nn_filtered(bin_types, open_bins, item_l, item_w, item_h, model, stats):
    """
    Same as HA3's pack_item, but every open bin is screened by the NN in a
    single batched call before geometric checks begin. Bins the NN rejects
    are skipped without any EP/orientation checking.
    """
    orientations = all_orientations(item_l, item_w, item_h)

    nn_predictions = nn_predict_all_bins(model, open_bins, item_l, item_w, item_h)
    stats["bins_screened"] += len(open_bins)

    for bin_obj, nn_fit in zip(open_bins, nn_predictions):

        if not nn_fit:
            stats["bins_rejected_by_nn"] += 1
            continue  # NN says skip this bin, try the next one

        # NN approved this bin - proceed with normal HA3 geometric packing
        # Since EP list is already sorted bottom-rear first, we just iterate in order
        for ep_x, ep_y, ep_z in bin_obj.ep_list:
            valid_orientations = []
            
            for (ol, ow, oh) in orientations:
                if can_place(ep_x, ep_y, ep_z, ol, ow, oh, 
                            bin_obj.length, bin_obj.width, bin_obj.height, 
                            bin_obj.placed_items):
                    valid_orientations.append((ol, ow, oh))
                    
            if valid_orientations:
                # We found a valid point! Pick the best orientation
                chosen_orient = valid_orientations[0]  # First valid orientation
                ol, ow, oh = chosen_orient
                
                # Place item
                bin_obj.placed_items.append((ep_x, ep_y, ep_z, ep_x + ol, ep_y + ow, ep_z + oh))
                
                # Update EPs
                bin_obj.ep_list = update_ep(bin_obj.ep_list, ep_x, ep_y, ep_z, ol, ow, oh, 
                                            bin_obj.placed_items, bin_obj.length, bin_obj.width, bin_obj.height)
                return bin_obj

        # NN approved the bin, but geometry disagreed - this happens
        # since the NN can't see individual EP shapes, only bin-level stats
        stats["nn_approved_but_geometry_failed"] += 1
        continue

    # No existing bin worked (NN-rejected or geometrically failed) - open a new one
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, ol, ow, oh,
                                            new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
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
    print(" RUNNING HA3 + NN FILTER")
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
    print(" HA3 + NN FILTER RESULTS")
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

    per_order_path = "ha3_nn_filtered_per_order.csv"
    with open(per_order_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "item_count", "bins_used", "cost", "utilization_pct", "time_seconds"])
        writer.writerows(per_order_rows)
    print(f"Saved: {per_order_path}")


if __name__ == "__main__":
    main()