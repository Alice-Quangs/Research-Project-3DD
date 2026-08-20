# generate_ha3_training_data.py
#
# Generates NN training data for HA3 (Extreme Points / EP-based heuristic).
# Runs the full BED-BPP dataset (all orders), logging every EP check
# made during packing as a Fit/No-Fit row.
#
# Does NOT modify ha3.py or bin_types.py - only imports from them.

import json
import csv
import time
from bin_types import bin_types
from ha3 import (
    Bin,
    all_orientations,
    can_place,
    update_ep,
)

DATA_PATH = "bed-bpp_v1.json"
OUTPUT_PATH = "ha3_training_data.csv"

HEADER = [
    "bin_length", "bin_width", "bin_height",
    "item_length", "item_width", "item_height",
    "red_cap", "fit"
]


def ha3_pack_with_logging(bin_types, open_bins, item_l, item_w, item_h, log_rows):
    """
    HA3 pack_item WITH logging.
    Logs EVERY EP in the bin:
        - If item fits at EP -> record fit=1 for the one chosen, fit=0 for others that fit but weren't chosen
        - If item DOES NOT fit at EP -> record fit=0 (invalid placement)
    """
    orientations = all_orientations(item_l, item_w, item_h)

    for bin_obj in open_bins:
        bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
        used_volume = sum((x2 - x1) * (y2 - y1) * (z2 - z1) for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items)
        red_cap = bin_volume - used_volume

        valid_ep_list = []
        valid_orientations_per_ep = []

        # First pass: find which EPs the item CAN fit at
        for ep_x, ep_y, ep_z in bin_obj.ep_list:
            valid_orientations = []
            for (ol, ow, oh) in orientations:
                if can_place(ep_x, ep_y, ep_z, ol, ow, oh, 
                            bin_obj.length, bin_obj.width, bin_obj.height, 
                            bin_obj.placed_items):
                    valid_orientations.append((ol, ow, oh))
            if valid_orientations:
                valid_ep_list.append((ep_x, ep_y, ep_z))
                valid_orientations_per_ep.append(valid_orientations)

        # Log ALL EPs - both valid and invalid
        for ep in bin_obj.ep_list:
            if ep in valid_ep_list:
                # This EP can fit the item
                # In HA3, the first valid EP is chosen (since ep_list is pre-sorted)
                if valid_ep_list:
                    chosen_ep = valid_ep_list[0]  # First valid EP
                    is_chosen = (ep == chosen_ep)

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
                    0  # No Fit at this EP
                ])

        # If there are valid EPs, place the item (same as original HA3)
        if valid_ep_list:
            chosen_ep = valid_ep_list[0]  # First valid EP (ep_list is pre-sorted)
            ep_index = valid_ep_list.index(chosen_ep)
            chosen_orient = valid_orientations_per_ep[ep_index][0]  # First valid orientation
            ol, ow, oh = chosen_orient
            ep_x, ep_y, ep_z = chosen_ep

            bin_obj.placed_items.append((ep_x, ep_y, ep_z, ep_x + ol, ep_y + ow, ep_z + oh))
            bin_obj.ep_list = update_ep(bin_obj.ep_list, ep_x, ep_y, ep_z, ol, ow, oh,
                                        bin_obj.placed_items, bin_obj.length, bin_obj.width, bin_obj.height)
            return bin_obj

    # No existing bin worked - open a new one
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, ol, ow, oh,
                                            new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
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
    print(" GENERATING HA3 TRAINING DATA (FULL DATASET)")
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
            ha3_pack_with_logging(bin_types, open_bins, item_l, item_w, item_h, log_rows)

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

    with open("ha3_generation_timing.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["orders_processed", "total_rows", "fit_count", "nofit_count", "generation_time_seconds"])
        writer.writerow([n_orders, len(log_rows), fit_count, nofit_count, elapsed])
    print("Saved: ha3_generation_timing.csv")


if __name__ == "__main__":
    main()