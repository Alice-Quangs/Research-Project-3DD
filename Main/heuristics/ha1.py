# %%

import os
import json
import numpy as np
import math
from bin_types import bin_types, Bin
# Note: visualise_result is imported further down, INSIDE run_f53, and only
# when visualize=True is actually requested. That way, just importing this
# file (e.g. from ha1_nn.py) never requires visualiser.py to exist.


# %%




# %%

def update_ems(ems_list, item_x, item_y, item_z, item_l, item_w, item_h):
    """
    This function updates the list of EMSs after placing an item in the bin
    item_x, item_y and item_z are the co-ordinates of the item's bottom left back corner
    item_l, item_w and item_h are the dimensions of the item

    """

    item_x_end = item_x + item_l
    item_y_end = item_y + item_w
    item_z_end = item_z + item_h


    new_ems_list = []

    for (ex, ey, ez, ex_end, ey_end, ez_end) in ems_list:

        # checking if the ems overlaps with the placed item at all, if no overlap then keep the item
        no_overlap = ( item_x_end <= ex or item_x >= ex_end or
                       item_y_end <= ey or item_y >= ey_end or
                       item_z_end <= ez or item_z >= ez_end   ) 
        
        if no_overlap:
            new_ems_list.append( (ex, ey, ez, ex_end, ey_end, ez_end)  )
            continue

        # If there is an overlap then split the sapce into 6 new EMSs

        # left of item ( along y )
        if ey < item_y:
            new_ems_list.append(  (ex, ey, ez, ex_end, item_y, ez_end) )


        # right of item ( along y  )
        if ey_end > item_y_end:
            new_ems_list.append(  (ex, item_y_end, ez, ex_end, ey_end, ez_end  )  )

        # top of item ( along z  )
        if ez_end > item_z_end:
            new_ems_list.append(  (ex, ey, item_z_end, ex_end, ey_end, ez_end) )

        # bottom of item ( along z  )
        if ez < item_z:
            new_ems_list.append(  (ex, ey, ez, ex_end, ey_end, item_z)  )

        # front of item ( along x  )
        if ex_end > item_x_end:
            new_ems_list.append(  (item_x_end, ey, ez, ex_end, ey_end, ez_end)   )

        # back of item ( along x  )
        if ex < item_x:
            new_ems_list.append(  (ex, ey, ez, item_x, ey_end, ez_end) )

    # Getting rid of EMSs that are completely contained within another EMS

    final_ems_list = []
    for (ex, ey, ez, ex_end, ey_end, ez_end) in new_ems_list:
        contained = False
        for (ex2, ey2, ez2, ex2_end, ey2_end, ez2_end) in new_ems_list:
            if (ex2, ey2, ez2, ex2_end, ey2_end, ez2_end) == (ex, ey, ez, ex_end, ey_end, ez_end):
                continue # skip checking against itself
            if (ex >= ex2 and ey >= ey2 and ez >= ez2 and ex_end <= ex2_end and ey_end <= ey2_end and ez_end <= ez2_end):
                contained = True
                break
        if not contained:
            final_ems_list.append( (ex, ey, ez, ex_end, ey_end, ez_end)  )

    return final_ems_list
        


# %%



# %%

def all_orientations(l, w, h):
    """
    The function returns all 6 different possible orientations of an item with dimensions l, w, h

    (x, y, z)

    """

    orientations = [(l, w, h), 
                    (l, h, w),
                    (w, l, h),
                    (w, h, l),
                    (h, l, w),
                    (h, w, l),
    ]


    return list(set(orientations))  # Removes duplicates if any


# %%



# %%


def fits_in_ems(ol, ow, oh, ex, ey, ez, ex_end, ey_end, ez_end):
    """
    Checks if an item with dimensions ol, ow, oh can fit in an EMS with coordinates ex, ey, ez, ex_end, ey_end, ez_end
    Returns True if it fits, False otherwise

    """

    does_it_fit = (ol <= (ex_end - ex) and ow <= (ey_end - ey) and oh <= (ez_end - ez))

    return does_it_fit


# %%


# %%

def best_ems(valid_ems_list, bin_l, bin_w, bin_h):
    """
    This function returns the best ems to place an item in, that means the EMS with the smallest x,
    then z, then closest to oe of the two rear-bottom corners of the bin.

    """
    min_x_ems = []
    min_z_ems = []

    # Find the EMSs with the smallest x coordinate
    min_x = min(ems[0] for ems in valid_ems_list)

    for ems in valid_ems_list:
        if ems[0] == min_x:
            min_x_ems.append(ems)

    if len(min_x_ems) == 1:
        return min_x_ems[0]

    # Find the EMSs with the smallest z coordinate among those with the smallest x coordinate        
    min_z = min(ems[2] for ems in min_x_ems)

    for ems in min_x_ems:
        if ems[2] == min_z:
            min_z_ems.append(ems)

    if len(min_z_ems) == 1:
        return min_z_ems[0]
    
    
    # Getting distance from the two rear-bottom corners of the bin for each of the EMSs that are still tied up

    distances = []

    coord_rear_bottom_left = (0, 0, 0)
    coord_rear_bottom_right = (0, bin_w, 0)
    
    for ems in min_z_ems:
        ems_coord = (ems[0], ems[1], ems[2], ems[3], ems[4], ems[5])
        d1_to_left_rear = math.sqrt( (ems_coord[3]  - coord_rear_bottom_left[0] )**2 +  (ems_coord[1]  -  coord_rear_bottom_left[1] )**2 + ( ems_coord[2]  -   coord_rear_bottom_left[2] )**2   )
        d1_to_right_rear = math.sqrt( (ems_coord[3]  - coord_rear_bottom_right[0] )**2 +  (ems_coord[1]  -  coord_rear_bottom_right[1] )**2 + ( ems_coord[2]  -   coord_rear_bottom_right[2] )**2   )
        d2_to_left_rear = math.sqrt( (ems_coord[3]  - coord_rear_bottom_left[0] )**2 +  (ems_coord[4]  -  coord_rear_bottom_left[1] )**2 + ( ems_coord[2]  -   coord_rear_bottom_left[2] )**2   )
        d2_to_right_rear = math.sqrt( (ems_coord[3]  - coord_rear_bottom_right[0] )**2 +  (ems_coord[4]  -  coord_rear_bottom_right[1] )**2 + ( ems_coord[2]  -   coord_rear_bottom_right[2] )**2   )
        min_distance = min(d1_to_left_rear, d1_to_right_rear, d2_to_left_rear, d2_to_right_rear)
        distances.append(min_distance)

    # Find the EMS with the smallest distance to either of the two rear-bottom corners of the bin
    min_distance_index = distances.index(min(distances))


    return min_z_ems[min_distance_index]


# %%


# %%

def best_orientation(valid_orientations_list):
    """
    This function returns the best orientation that maximises the base area with extrusion in the x-axis

    """

    base_areas = [ol * ow for (ol, ow, oh) in valid_orientations_list]
    max_base_area = max(base_areas)
    
    max_area_orient = []

    for idx, area in enumerate(base_areas):
        if area == max_base_area:
            max_area_orient.append(valid_orientations_list[idx])

    if len(max_area_orient) == 1:
        return max_area_orient[0]
    
    # Then take the orientation with greatest extrusion in the x-axis (length)
    max_length = max(ol for (ol, ow, oh) in max_area_orient)

    final_orient = []

    for idx, orient in enumerate(max_area_orient):
        if orient[0] == max_length:
            final_orient.append(orient)

    
    return final_orient[0]


# %%



# %%

def can_pack(bin_obj, item):
    """Check EMS dimensional feasibility without modifying bin state."""
    orientations = all_orientations(item['length'], item['width'], item['height'])
    return any(
        fits_in_ems(*orientation, *ems)
        for ems in bin_obj.ems_list
        for orientation in orientations
    )


def pack_item(bin_types, open_bins, item_l, item_w, item_h):
    """
    Packs an item into the first open bin that fits (First-Fit).
    If no open bin works, opens the smallest new bin that can fit the item.
    """

    orientations = all_orientations(item_l, item_w, item_h)

    # Trying existing open bins
    for bin in open_bins:

        valid_ems_list = []
        valid_orientations_per_ems_list = []

        for ems in bin.ems_list:

            valid_orientations = []

            for (ol, ow, oh) in orientations:

                ex     = ems[0]
                ey     = ems[1]
                ez     = ems[2]
                ex_end = ems[3]
                ey_end = ems[4]
                ez_end = ems[5]

                if fits_in_ems(ol, ow, oh, ex, ey, ez, ex_end, ey_end, ez_end):
                    valid_orientations.append((ol, ow, oh))

            if valid_orientations:
                valid_ems_list.append(ems)
                valid_orientations_per_ems_list.append(valid_orientations)

        if valid_ems_list:

            chosen_ems    = best_ems(valid_ems_list, bin.length, bin.width, bin.height)
            ems_index     = valid_ems_list.index(chosen_ems)
            chosen_orient = best_orientation(valid_orientations_per_ems_list[ems_index])

            item_x = chosen_ems[0]
            item_y = chosen_ems[1]
            item_z = chosen_ems[2]

            ol = chosen_orient[0]
            ow = chosen_orient[1]
            oh = chosen_orient[2]

            bin.placed_items.append((item_x, item_y, item_z, item_x + ol, item_y + ow, item_z + oh))
            bin.ems_list = update_ems(bin.ems_list, item_x, item_y, item_z, ol, ow, oh)

            bin.register_placement(ol, ow, oh)
            return bin

    # No open bin worked, so we open the smallest new bin that fits
    for bin_type in bin_types:

        fits = False

        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                fits = True
                best_ol = ol
                best_ow = ow
                best_oh = oh
                break

        if fits:
            new_bin = Bin(bin_type)
            new_bin.placed_items.append((0, 0, 0, best_ol, best_ow, best_oh))
            new_bin.ems_list = update_ems(new_bin.ems_list, 0, 0, 0, best_ol, best_ow, best_oh)
            new_bin.register_placement(best_ol, best_ow, best_oh)
            open_bins.append(new_bin)
            return new_bin

    # If nothing fits at all
    print("Item could not be packed into any bin type!")
    return None




# %%




# %%


def run_f53(sequence, bin_types, visualize=True, save_path="f53_result.png"):
    """
    This function runs the F53 heuristic on a given sequence of items and returns the open bins,
    the number of bins used, and the total cost.

    """

    open_bins = []
    items_processed = 0

    for item in sequence:
        item_l = item['length']
        item_w = item['width']
        item_h = item['height']

        pack_item(bin_types, open_bins, item_l, item_w, item_h)
        items_processed += 1

    total_cost = sum(bin.cost for bin in open_bins)
    num_bins_used = len(open_bins)

    if visualize:
        from visualiser import visualise_result  # only needed here, only when actually visualizing
        print_bin_summary(open_bins, items_processed)
        if open_bins:
            print_items_in_bin(open_bins[0], 1)
            visualise_result(open_bins, save_path=save_path)

    return open_bins, num_bins_used, total_cost


# %%



# %%

def print_bin_summary(open_bins, sequence_length=None):
    """
    Prints a detailed summary of all bins after packing.
    
        open_bins: List of Bin objects
        sequence_length: Total number of items in the sequence (optional)
    """
    print("\n" + "=" * 70)
    print(" BIN PACKING SUMMARY")
    print("=" * 70)
    
    if sequence_length:
        print(f" Total items in sequence: {sequence_length}")
    
    print(f" Total bins used: {len(open_bins)}")
    print("=" * 70)
    
    total_volume_used = 0
    total_bin_volume = 0
    
    for i, bin_obj in enumerate(open_bins, 1):
        print(f"\n Bin {i}: {bin_obj.name}")
        print(f"   {'─' * 50}")
        print(f"    Dimensions: {bin_obj.length} x {bin_obj.width} x {bin_obj.height}")
        print(f"    Items placed: {len(bin_obj.placed_items)}")
        
        # Calculate bin volume
        bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
        total_bin_volume += bin_volume
        
        # Calculate used volume
        used_volume = 0
        for item in bin_obj.placed_items:
            x_start, y_start, z_start, x_end, y_end, z_end = item
            item_volume = (x_end - x_start) * (y_end - y_start) * (z_end - z_start)
            used_volume += item_volume
        
        total_volume_used += used_volume
        remaining_volume = bin_volume - used_volume
        utilization = (used_volume / bin_volume) * 100 if bin_volume > 0 else 0
        
        print(f"    Volume used: {used_volume:,} / {bin_volume:,} units³")
        print(f"    Remaining volume: {remaining_volume:,} units³")
        print(f"    Space utilization: {utilization:.1f}%")
        print(f"    Bin cost: {bin_obj.cost}")
    
    print("\n" + "=" * 70)
    total_utilization = (total_volume_used / total_bin_volume) * 100 if total_bin_volume > 0 else 0
    print(f" OVERALL STATISTICS:")
    print(f"   Total bins: {len(open_bins)}")
    print(f"   Total volume used: {total_volume_used:,} units³")
    print(f"   Total bin volume: {total_bin_volume:,} units³")
    print(f"   Overall space utilization: {total_utilization:.1f}%")
    print(f"   Total cost: {sum(bin.cost for bin in open_bins)}")
    print("=" * 70)

# %%




# %%


def print_items_in_bin(bin_obj, bin_number=None):
    """
    Prints details of all items in a specific bin.
    """
    if bin_number is None:
        bin_number = "?"
    
    print(f"\n Items in Bin {bin_number} ({bin_obj.name}):")
    print("   " + "-" * 55)
    print(f"   {'#':<4} {'Dimensions (LxWxH)':<18} {'Volume':<12} {'Position (x,y,z)':<20}")
    print("   " + "-" * 55)
    
    for idx, item in enumerate(bin_obj.placed_items, 1):
        x_start, y_start, z_start, x_end, y_end, z_end = item
        width = x_end - x_start
        depth = y_end - y_start
        height = z_end - z_start
        volume = width * depth * height
        print(f"   {idx:<4} {width:>3} x {depth:>3} x {height:>3}   {volume:>9,}   ({x_start:>3}, {y_start:>3}, {z_start:>3})")
    
    print("   " + "-" * 55)

# %%





# %%

if __name__ == "__main__":
    # Create Visuals folder if it doesn't exist
    os.makedirs("Visuals", exist_ok=True)

    # Load one order (order_index=0, i.e. the first one) straight from the
    # raw JSON dataset - no separate loader file needed.
    with open("../dataset/bed-bpp_v1.json", "r") as f:
        bed_bpp_data = json.load(f)

    first_order_id = list(bed_bpp_data.keys())[0]
    order = bed_bpp_data[first_order_id]

    sequence = []
    for key in sorted(order["item_sequence"].keys(), key=lambda x: int(x)):
        item = order["item_sequence"][key]
        sequence.append({
            "length": item["length/mm"],
            "width": item["width/mm"],
            "height": item["height/mm"],
        })

    open_bins, num_bins, total_cost = run_f53(sequence, bin_types, save_path="Visuals/ha1_visualisation.png")
    
    print(f"Bins used: {num_bins}")
    print(f"Total cost: {total_cost}")

# %%