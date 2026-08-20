# generate_ha2_training_data.py
#
# Generates NN training data for HA2 (F73 / EMS-based heuristic).
# Runs the full BED-BPP dataset (all orders), logging every EMS check
# made during packing as a Fit/No-Fit row.
#
# Does NOT modify ha2.py or bin_types.py - only imports from them.

import json
import csv
import time
from bin_types import bin_types, Bin
from ha2 import (
    all_orientations,
    fits_in_ems,
    update_ems,
    best_ems,
    best_orientation,
)

DATA_PATH = "bed-bpp_v1.json"
OUTPUT_PATH = "ha2_training_data.csv"

HEADER = [
    "bin_length", "bin_width", "bin_height",
    "item_length", "item_width", "item_height",
    "red_cap", "fit"
]


def ha2_pack_with_logging(bin_types, open_bins, item_l, item_w, item_h, log_rows):
    """
    HA2 pack_item WITH logging.
    Logs EVERY EMS in the bin:
        - If item fits in EMS -> record fit=1 for the one chosen, fit=0 for others that fit but weren't chosen
        - If item DOES NOT fit in EMS -> record fit=0 (invalid placement)
    """
    orientations = all_orientations(item_l, item_w, item_h)

    for bin_obj in open_bins:
        bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
        used_volume = sum((x2 - x1) * (y2 - y1) * (z2 - z1) for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items)
        red_cap = bin_volume - used_volume

        valid_ems_list = []
        valid_orientations_per_ems = []

        # First pass: find which EMSs the item CAN fit in
        for ems in bin_obj.ems_list:
            valid_orientations = []
            for (ol, ow, oh) in orientations:
                if fits_in_ems(ol, ow, oh, ems[0], ems[1], ems[2], ems[3], ems[4], ems[5]):
                    valid_orientations.append((ol, ow, oh))
            if valid_orientations:
                valid_ems_list.append(ems)
                valid_orientations_per_ems.append(valid_orientations)

        # Log ALL EMSs - both valid and invalid
        for ems in bin_obj.ems_list:
            if ems in valid_ems_list:
                if valid_ems_list:
                    chosen_ems = best_ems(valid_ems_list, bin_obj.length, bin_obj.width, bin_obj.height)
                    is_chosen = (ems == chosen_ems)

                    log_rows.append([
                        bin_obj.length, bin_obj.width, bin_obj.height,
                        item_l, item_w, item_h,
                        red_cap,
                        1 if is_chosen else 0
                    ])
            else:
                log_rows.append([
                    bin_obj.length, bin_obj.width, bin_obj.height,
                    item_l, item_w, item_h,
                    red_cap,
                    0  # No Fit
                ])

        # If there are valid EMSs, place the item (same as original)
        if valid_ems_list:
            chosen_ems = best_ems(valid_ems_list, bin_obj.length, bin_obj.width, bin_obj.height)
            ems_index = valid_ems_list.index(chosen_ems)
            chosen_orient = best_orientation(valid_orientations_per_ems[ems_index])
            ol, ow, oh = chosen_orient
            item_x, item_y, item_z = chosen_ems[0], chosen_ems[1], chosen_ems[2]

            bin_obj.placed_items.append((item_x, item_y, item_z, item_x + ol, item_y + ow, item_z + oh))
            bin_obj.ems_list = update_ems(bin_obj.ems_list, item_x, item_y, item_z, ol, ow, oh)
            return bin_obj

    # No existing bin worked - open a new one
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ems_list = update_ems(new_bin.ems_list, 0, 0, 0, ol, ow, oh)
                open_bins.append(new_bin)

                bin_volume = new_bin.length * new_bin.width * new_bin.height
                used_volume = sum((x2 - x1) * (y2 - y1) * (z2 - z1) for x1, y1, z1, x2, y2, z2 in new_bin.placed_items)
                red_cap = bin_volume - used_volume

                log_rows.append([
                    new_bin.length, new_bin.width, new_bin.height,
                    item_l, item_w, item_h,
                    red_cap,
                    1  # First item in new bin always fits
                ])

                return new_bin

    print("Item could not be packed!")
    return None


def main():
    print("Loading BED-BPP dataset...")
    with open(DATA_PATH, 'r') as f:
        dataset = json.load(f)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    print(f"Loaded {n_orders} orders")

    print("\n" + "=" * 60)
    print(" GENERATING HA2 TRAINING DATA (FULL DATASET)")
    print("=" * 60)

    log_rows = []

    start_time = time.perf_counter()

    for idx, order_id in enumerate(order_ids):
        order = dataset[order_id]

        items = []
        for key in sorted(order['item_sequence'].keys(), key=lambda x: int(x)):
            item = order['item_sequence'][key]
            items.append((item['length/mm'], item['width/mm'], item['height/mm']))

        open_bins = []

        for item_l, item_w, item_h in items:
            ha2_pack_with_logging(bin_types, open_bins, item_l, item_w, item_h, log_rows)

        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            print(f"  Processed {idx + 1}/{n_orders} orders, {len(log_rows)} rows so far")

    elapsed = time.perf_counter() - start_time

    print(f"\nTotal rows collected: {len(log_rows)}")
    print(f"Generation time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    # Write CSV
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(log_rows)

    print(f"Written to {OUTPUT_PATH}")

    # Summary
    fit_count = sum(1 for row in log_rows if row[7] == 1)
    nofit_count = sum(1 for row in log_rows if row[7] == 0)

    print("\n" + "=" * 60)
    print(" DATA SUMMARY")
    print("=" * 60)
    print(f"Fit: {fit_count}")
    print(f"No Fit: {nofit_count}")
    print(f"Total: {len(log_rows)}")
    print(f"Class balance: {fit_count / len(log_rows) * 100:.1f}% Fit / {nofit_count / len(log_rows) * 100:.1f}% No Fit")

    with open("ha2_generation_timing.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["orders_processed", "total_rows", "fit_count", "nofit_count", "generation_time_seconds"])
        writer.writerow([n_orders, len(log_rows), fit_count, nofit_count, elapsed])
    print("Saved: ha2_generation_timing.csv")


if __name__ == "__main__":
    main()