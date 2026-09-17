# bin_types.py

"""
Bin type definitions and Bin class for 3D bin packing.
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pprint
import os

os.makedirs('Visuals', exist_ok=True)

print("Loading dataset...")
with open("../dataset/bed-bpp_v1.json", "r") as f:
    bed_bpp_data = json.load(f)

items_data = []
for order in bed_bpp_data.values():
    for item in order["item_sequence"].values():
        items_data.append([
            item["length/mm"], item["width/mm"], item["height/mm"], item["weight/kg"]
        ])

df = pd.DataFrame(items_data, columns=['length', 'width', 'height', 'weight'])

print("Running K-Means clustering (k=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df[['length', 'width', 'height']])

# Keep the original cluster ID as a column so we can map colors correctly
bins = df.groupby('cluster').max().reset_index()
bins['volume'] = bins['length'] * bins['width'] * bins['height']

# Sort by volume to assign the correct names from smallest to largest
bins = bins.sort_values('volume').reset_index(drop=True)
bins['bin_name'] = ["Small", "Medium", "Large", "Extra Large"]

# W, H, D are the length, width, height of the largest bin (unscaled),
# used as the reference dimensions in the cost formula below
largest_bin = bins.loc[bins['volume'].idxmax()]
W, H, D = largest_bin['length'], largest_bin['width'], largest_bin['height']
WHD = W * H * D

# c_k = ceil(1000 * (1.2*V_k - 0.2*W*H*D) / (W*H*D))
bins['cost'] = np.ceil(1000 * (1.2 * bins['volume'] - 0.2 * WHD) / WHD).astype(int)

bin_types = []
for i, row in bins.iterrows():
    bin_types.append({
        "name": row['bin_name'].lower().replace(" ", "_"),
        "length": int(row['length'] * 1.05),
        "width": int(row['width'] * 1.05),
        "height": int(row['height'] * 1.05),
        "cost": int(row['cost']),
        "max_weight": int(row['weight'] * 1.05)
    })

print("\n=== COPY AND PASTE THIS INTO online_generator.py ===")
print("bin_types = \\")
pprint.pprint(bin_types, sort_dicts=False)


if __name__ == "__main__":
    print("\nGenerating 3D visualization of the clusters...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster individually to build a proper legend
    for i, row in bins.iterrows():
        cluster_id = row['cluster']
        bin_name = row['bin_name']
        
        # Filter the dataframe for items belonging to this specific bin size
        cluster_items = df[df['cluster'] == cluster_id]
        
        # Scatter the items and label them for the legend
        ax.scatter(cluster_items['length'], cluster_items['width'], cluster_items['height'], 
                   label=f'{bin_name} Items', alpha=0.4)
        
        # Plot the red star for this bin
        ax.scatter(row['length'], row['width'], row['height'], 
                   c='red', marker='*', s=300, edgecolor='black')
        
        # Add floating text above the star
        ax.text(row['length'], row['width'], row['height'] + 40,  
                bin_name, size=11, zorder=1, color='black', fontweight='bold')

    ax.set_title('K-Means Clustering of BED-BPP Grocery Items')
    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Width (mm)')
    ax.set_zlabel('Height (mm)')

    # Add a custom legend entry for the stars
    ax.scatter([], [], [], c='red', marker='*', s=200, edgecolor='black', label='Max Box Dimensions')
    ax.legend()

    plt.savefig('Visuals/cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


# 
# BIN CLASS
# 

class Bin:
    """
    Represents a bin's dimensions and its available Empty Maximal Spaces (EMS)
    """
    def __init__(self, bin_type):
        self.name = bin_type["name"]
        self.length = bin_type["length"]
        self.width = bin_type["width"]
        self.height = bin_type["height"]
        self.cost = bin_type["cost"]

        # The first ems is the entire bin space
        self.ems_list = [(0, 0, 0, self.length, self.width, self.height)]

        # Keeping track of items placed inside
        self.placed_items = []

        # --- Incrementally-tracked stats -----------------------------
        # These used to be recomputed from scratch (rescanning
        # placed_items / ems_list) every single time the NN gatekeeper
        # screened this bin - which happens many times per bin, since
        # every open bin gets screened for every incoming item whether
        # or not anything ends up placed in it. That made feature-
        # building roughly as expensive as the geometry check it was
        # meant to replace.
        #
        # Instead, we cache them here and only refresh them right after
        # an item is ACTUALLY placed in this bin (see refresh_ems_stats
        # below), since that's the only event that can change them.
        # Screening a bin and rejecting it no longer costs anything
        # beyond reading these attributes.
        self.volume = self.length * self.width * self.height
        self.used_volume = 0
        self.refresh_ems_stats()

    @property
    def red_cap(self):
        """Residual (remaining) capacity - O(1) read instead of
        re-summing every entry in placed_items on every call."""
        return self.volume - self.used_volume

    def refresh_ems_stats(self):
        """
        Rescans self.ems_list ONCE and caches the max/min EMS volume,
        their dimensions, and the EMS count. Call this only right after
        ems_list has actually changed (i.e. right after a real item
        placement) - not on every screening attempt.
        """
        if not self.ems_list:
            self.max_ems_volume = 0
            self.max_ems_length = self.max_ems_width = self.max_ems_height = 0
            self.min_ems_volume = 0
            self.min_ems_length = self.min_ems_width = self.min_ems_height = 0
            self.ems_count = 0
            return

        volumes = []
        dims = []
        for (x1, y1, z1, x2, y2, z2) in self.ems_list:
            l, w, h = x2 - x1, y2 - y1, z2 - z1
            volumes.append(l * w * h)
            dims.append((l, w, h))

        max_i = volumes.index(max(volumes))
        min_i = volumes.index(min(volumes))

        self.max_ems_volume = volumes[max_i]
        self.max_ems_length, self.max_ems_width, self.max_ems_height = dims[max_i]
        self.min_ems_volume = volumes[min_i]
        self.min_ems_length, self.min_ems_width, self.min_ems_height = dims[min_i]
        self.ems_count = len(self.ems_list)

    def register_placement(self, ol, ow, oh):
        """
        Call this right after appending to placed_items and reassigning
        ems_list (via update_ems), so used_volume and the cached EMS
        stats stay in sync with what actually happened. Kept as a single
        call so every packing function updates these the same way rather
        than each remembering to do it by hand in slightly different
        order.
        """
        self.used_volume += ol * ow * oh
        self.refresh_ems_stats()

    def __repr__(self):
        return f"Bin({self.name}, EMSs: {len(self.ems_list)}, Items: {len(self.placed_items)})"