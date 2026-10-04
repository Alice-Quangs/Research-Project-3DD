import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pprint

print("Loading dataset...")
with open('bed-bpp_v1.json', 'r') as f:
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

base_cost = 10
min_vol = bins['volume'].min()
bins['cost'] = np.ceil((bins['volume'] / min_vol) * base_cost).astype(int)

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

plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()