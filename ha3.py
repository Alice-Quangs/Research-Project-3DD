# %%
import os
import math
import numpy as np
from bin_types import bin_types
from bed_loader import load_bed_sequence
from visualiser import visualise_result



# %%
class Bin:
    """
    Represents a bin's dimensions and its available Extreme Points (EP)
    """
    def __init__(self, bin_type):
        self.name = bin_type["name"]
        self.length = bin_type["length"]
        self.width = bin_type["width"]
        self.height = bin_type["height"]
        self.cost = bin_type["cost"]

        # The first extreme point is the bottom-rear-left corner of the empty bin
        self.ep_list = [(0, 0, 0)]

        # Keeping track of items placed inside
        self.placed_items = []
    
    def __repr__(self):
        return f"Bin({self.name}, EPs: {len(self.ep_list)}, Items: {len(self.placed_items)})"

# %%
def update_ep(ep_list, item_x, item_y, item_z, ol, ow, oh, placed_items, bin_l, bin_w, bin_h):
    """
    Updates the list of Extreme Points after placing an item.
    Generates new points at the upper, left-front, and right-rear corners.
    """
    # Remove the point that was just used (if it exists)
    if (item_x, item_y, item_z) in ep_list:
        ep_list.remove((item_x, item_y, item_z))
    
    # Generate 3 new candidate extreme points based on the placed item
    new_points = [
        (item_x + ol, item_y, item_z), # Left-front (extrusion along X)
        (item_x, item_y + ow, item_z), # Right-rear (extrusion along Y)
        (item_x, item_y, item_z + oh)  # Upper (extrusion along Z)
    ]
    
    for px, py, pz in new_points:
        # 1. Discard if the point is strictly outside the bin boundaries
        if px >= bin_l or py >= bin_w or pz >= bin_h:
            continue
            
        # 2. Discard if the point is strictly inside any previously placed item
        is_inside = False
        for (ix, iy, iz, ix_end, iy_end, iz_end) in placed_items:
            # A point exactly on the face/edge of a placed item is valid, 
            # so we use strictly less than (<) and strictly greater than (>)
            if (ix < px < ix_end) and (iy < py < iy_end) and (iz < pz < iz_end):
                is_inside = True
                break
                
        if not is_inside and (px, py, pz) not in ep_list:
            ep_list.append((px, py, pz))
            
    # Sort points to prioritize Bottom (Z), then Rear (X), then Left (Y)
    ep_list.sort(key=lambda p: (p[2], p[0], p[1]))
    
    return ep_list

# %%
def all_orientations(l, w, h):
    """
    Returns all 6 different possible orientations of an item with dimensions l, w, h
    """
    orientations = [
        (l, w, h), 
        (l, h, w),
        (w, l, h),
        (w, h, l),
        (h, l, w),
        (h, w, l)
    ]
    return list(set(orientations))

# %%
def can_place(ep_x, ep_y, ep_z, ol, ow, oh, bin_l, bin_w, bin_h, placed_items):
    """
    Checks if an item of dimension (ol, ow, oh) can be placed at point (ep_x, ep_y, ep_z)
    without exceeding bin dimensions or overlapping with existing items.
    """
    # 1. Check bin boundaries
    if ep_x + ol > bin_l or ep_y + ow > bin_w or ep_z + oh > bin_h:
        return False
        
    # 2. Check for collisions with existing items
    for (ix, iy, iz, ix_end, iy_end, iz_end) in placed_items:
        no_overlap = (ep_x + ol <= ix or ep_x >= ix_end or
                      ep_y + ow <= iy or ep_y >= iy_end or
                      ep_z + oh <= iz or ep_z >= iz_end)
        if not no_overlap:
            return False # Collision detected
            
    return True

# %%
def best_orientation(valid_orientations_list):
    """
    Returns the best orientation that maximises the base area,
    breaking ties with extrusion in the x-axis.
    """
    base_areas = [ol * ow for (ol, ow, oh) in valid_orientations_list]
    max_base_area = max(base_areas)
    
    max_area_orient = [orient for idx, orient in enumerate(valid_orientations_list) if base_areas[idx] == max_base_area]

    if len(max_area_orient) == 1:
        return max_area_orient[0]
    
    # Take orientation with greatest extrusion in the x-axis (length)
    max_length = max(ol for (ol, ow, oh) in max_area_orient)
    final_orient = [orient for orient in max_area_orient if orient[0] == max_length]
    
    return final_orient[0]

# %%
def pack_item(bin_types, open_bins, item_l, item_w, item_h):
    """
    Packs an item into the first open bin that fits using Extreme Points.
    If no open bin works, opens the smallest new bin that can fit the item.
    """
    orientations = all_orientations(item_l, item_w, item_h)

    # Trying existing open bins
    for bin_obj in open_bins:
        # Since EP list is already sorted bottom-rear first, we just iterate in order
        for ep_x, ep_y, ep_z in bin_obj.ep_list:
            valid_orientations = []
            
            for (ol, ow, oh) in orientations:
                if can_place(ep_x, ep_y, ep_z, ol, ow, oh, bin_obj.length, bin_obj.width, bin_obj.height, bin_obj.placed_items):
                    valid_orientations.append((ol, ow, oh))
                    
            if valid_orientations:
                # We found a valid point! Pick the best orientation
                chosen_orient = best_orientation(valid_orientations)
                ol, ow, oh = chosen_orient
                
                # Place item
                bin_obj.placed_items.append((ep_x, ep_y, ep_z, ep_x + ol, ep_y + ow, ep_z + oh))
                
                # Update EPs
                bin_obj.ep_list = update_ep(bin_obj.ep_list, ep_x, ep_y, ep_z, ol, ow, oh, 
                                            bin_obj.placed_items, bin_obj.length, bin_obj.width, bin_obj.height)
                return bin_obj

    # No open bin worked, so we open the smallest new bin that fits
    for bin_type in bin_types:
        fits = False
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                fits = True
                best_ol, best_ow, best_oh = ol, ow, oh
                break

        if fits:
            new_bin = Bin(bin_type)
            new_bin.placed_items.append((0, 0, 0, best_ol, best_ow, best_oh))
            new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, best_ol, best_ow, best_oh, 
                                        new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
            open_bins.append(new_bin)
            return new_bin

    print("Item could not be packed into any bin type!")
    return None

# %%
def run_ep_method(sequence, bin_types, visualize=True, save_path="ep_result.png"):
    """
    Runs the Extreme Points packing logic on a given sequence of items.
    """
    open_bins = []
    items_processed = 0

    for item in sequence:
        item_l = item['length']
        item_w = item['width']
        item_h = item['height']
        pack_item(bin_types, open_bins, item_l, item_w, item_h)
        items_processed += 1

    total_cost = sum(bin_obj.cost for bin_obj in open_bins)
    num_bins_used = len(open_bins)

    if visualize:
        print_bin_summary(open_bins, items_processed)
        if open_bins:
            print_items_in_bin(open_bins[0], 1)
            visualise_result(open_bins, save_path=save_path)

    return open_bins, num_bins_used, total_cost

# %%
def print_bin_summary(open_bins, sequence_length=None):
    # (Kept identical to your original code)
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
        
        bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
        total_bin_volume += bin_volume
        
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
def print_items_in_bin(bin_obj, bin_number=None):
    # (Kept identical to your original code)
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
if __name__ == "__main__":
    os.makedirs("Visuals", exist_ok=True)
    
    sequence = load_bed_sequence("bed-bpp_v1.json", order_index=0)
    
    open_bins, num_bins, total_cost = run_ep_method(sequence, bin_types, save_path="Visuals/ha3_visualisation.png")
    
    print(f"Bins used: {num_bins}")
    print(f"Total cost: {total_cost}")