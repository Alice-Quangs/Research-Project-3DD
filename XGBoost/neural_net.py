

# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ems_heuristic


# %%










# %%

############## VOXEL REPRESENTATION ############
# The codes representing the voxel states
OCCUPIED = 1 
EMPTY_SUPPORTED = 2 # empty voxel with something beneath it so an item may be placed
EMPTY_UNSUPPORTED = 3 # empty floating voxel

# %%


















# %%

def bin_to_voxel_grid(bin_obj, resolution=10):
    """
    Converts a Bin object (with .length, .width, .height, .placed_items) into
    a fixed-size "resolution x resolution x resolution" voxel grid.

    """

    R = resolution

    voxel_size_x = bin_obj.length / R
    voxel_size_y = bin_obj.width / R
    voxel_size_z = bin_obj.height / R

    grid = np.zeros((R, R, R), dtype=np.int8) # gives R x R x R grid of zeros

    # PASS 1: marking occupied voxels
    for (x_start, y_start, z_start, x_end, y_end, z_end) in bin_obj.placed_items:

        # flooring the start co-ords
        ix_start = int(np.floor(x_start / voxel_size_x)) # how many voxel widths fit between the origin and the item's start edge
        iy_start = int(np.floor(y_start / voxel_size_y))
        iz_start = int(np.floor(z_start / voxel_size_z))

        # ceiling the end co-ords
        ix_end = int(np.ceil(x_end / voxel_size_x))
        iy_end = int(np.ceil(y_end / voxel_size_y))
        iz_end = int(np.ceil(z_end / voxel_size_z))

        ix_start, ix_end = max(0, ix_start), min(R, ix_end)
        iy_start, iy_end = max(0, iy_start), min(R, iy_end)
        iz_start, iz_end = max(0, iz_start), min(R, iz_end)

        grid[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] = OCCUPIED

    # PASS 2: resolve support for remaining empty voxels
    for ix in range(R):
        for iy in range(R):
            for iz in range(R):
                if grid[ix, iy, iz] == OCCUPIED:
                    continue
                if iz == 0 or grid[ix, iy, iz - 1] == OCCUPIED:
                    grid[ix, iy, iz] = EMPTY_SUPPORTED
                else:
                    grid[ix, iy, iz] = EMPTY_UNSUPPORTED

    return grid

# %%













# %%

def voxel_grid_to_onehot(grid):
    """
    Converts the integer-coded voxel grid (values 1, 2, 3) into a one-hot
    encoded array of shape (3, R, R, R), suitable as NN input.
    Channel 0 = occupied, channel 1 = empty supported, channel 2 = empty unsupported.
    """
    R = grid.shape[0]
    onehot = np.zeros((3, R, R, R), dtype=np.float32)
    onehot[0] = (grid == OCCUPIED)
    onehot[1] = (grid == EMPTY_SUPPORTED)
    onehot[2] = (grid == EMPTY_UNSUPPORTED)
    return onehot

# %%














############# LABEL GENERATION ############

# Design decision: for every item arrival, we look at
# ALL currently open bins that can physically fit the item. Each becomes a
# candidate. We compute what the bin's volume utilisation would become if
# the item were placed there. Candidates at or above the median projected
# utilisation are labelled "good" (1), the rest "bad" (0). Bins the item
# doesn't physically fit into are simply not included as training samples,
# since the NN's job is to rank/filter feasible candidates, not detect
# infeasibility (that's a simple geometry check, not something worth
# learning).









# %%

def item_fits_in_bin(item_dims, bin_obj):
    item_l, item_w, item_h = item_dims
    orientations = ems_heuristic.all_orientations(item_l, item_w, item_h)
    for ems in bin_obj.ems_list:
        ex, ey, ez, ex_end, ey_end, ez_end = ems
        for (ol, ow, oh) in orientations:
            if ems_heuristic.fits_in_ems(ol, ow, oh, ex, ey, ez, ex_end, ey_end, ez_end):
                return True
    return False
# %%











# %%

def bin_used_volume(bin_obj):
    total = 0
    for (xs, ys, zs, xe, ye, ze) in bin_obj.placed_items:
        total += (xe - xs) * (ye - ys) * (ze - zs)
    return total
# %%










# %%
def bin_volume(bin_obj):
    return bin_obj.length * bin_obj.width * bin_obj.height

# %%












# %%

def build_training_data(sequences, bin_types, resolution=10, max_sequences=None):
    """
    Builds NN training data by simulating each sequence item-by-item.

    For each item, every open bin it could fit into becomes a candidate,
    labelled 'good' or 'bad' relative to the other candidates at that moment
    (based on projected cost efficiency). The item is then packed with F53 so
    bins evolve as they would in a real run.

    Returns a list of (voxel_onehot, item_dims, label) samples.
    """

    samples = []
    seqs = sequences if max_sequences is None else sequences[:max_sequences]

    for sequence in seqs:
        open_bins = []

        for item in sequence:
            item_l, item_w, item_h = item[0], item[1], item[2]
            item_dims = (item_l, item_w, item_h)

            candidates = [] # list of (bin, hypothetical_cost_efficiency) pairs
            for bin_obj in open_bins: 
                if item_fits_in_bin(item_dims, bin_obj):
                    projected_used = bin_used_volume(bin_obj) + (item_l * item_w * item_h)
                    projected_util = projected_used / bin_volume(bin_obj)
                    
                    # Calculate cost efficiency instead of just spatial utilization
                    cost_efficiency = projected_util / bin_obj.cost
                    candidates.append((bin_obj, cost_efficiency))

            if candidates:
                # Calculate the median of cost efficiencies across all candidate bins for the item
                median_eff = float(np.median([eff for _, eff in candidates])) 
                for bin_obj, eff in candidates:
                    # Label is 1.0 if the bin's cost efficiency is above or equal to the median
                    label = 1.0 if eff >= median_eff else 0.0
                    onehot = voxel_grid_to_onehot(bin_to_voxel_grid(bin_obj, resolution))
                    samples.append((onehot, np.array(item_dims, dtype=np.float32), bin_obj.cost, label))

            # Actually place the item so the bins evolve realistically for the next item
            ems_heuristic.pack_item(bin_types, open_bins, item_l, item_w, item_h)

    return samples
# %%












# %%

############## DATASET #############

class BinFilterDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        onehot, item_dims, bin_cost, label = self.samples[idx]
        return (
            torch.tensor(onehot, dtype=torch.float32),
            torch.tensor(item_dims, dtype=torch.float32),
            torch.tensor([bin_cost], dtype=torch.float32), # Added bin_cost
            torch.tensor(label, dtype=torch.float32),
        )

# %%















# %%

############# MODEL #############


class BinFilterNet(nn.Module):
    """
    Single hidden-layer NN: takes the
    flattened voxel grid plus the item's dimensions AND the bin's cost, and outputs a single
    sigmoid score for "how good a candidate is this bin".
    """

    def __init__(self, resolution=10, hidden_size=128):
        super().__init__()
        voxel_input_size = 3 * resolution ** 3
        item_input_size = 4  # Changed from 3 to 4 to include bin_cost

        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(voxel_input_size + item_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size // 2, 1)

    def forward(self, voxel_grid, item_dims, bin_cost):
        voxel_flat = self.flatten(voxel_grid)
        # Concatenate voxel grid, item dimensions, and the bin cost
        combined = torch.cat([voxel_flat, item_dims, bin_cost], dim=1)
        hidden = self.hidden(combined)
        score = torch.sigmoid(self.output(hidden))
        return score
    
# %%
# 
# 














# %%
############ TRAINING #############


def train_model(samples, resolution=10, epochs=100, batch_size=32, lr=1e-3, min_delta=1e-4, patience=5):
    
    """
    Trains the model with early stopping.
 
    epochs      - upper bound on training passes (a safety cap, not a target)
    min_delta   - the smallest loss improvement that still counts as "progress"
    patience    - how many epochs in a row we tolerate without a min_delta
                  improvement before we stop early
 
    Stops as soon as the loss hasn't meaningfully improved for `patience`
    consecutive epochs, or once `epochs` is reached, whichever comes first.
    """
    
    dataset = BinFilterDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BinFilterNet(resolution=resolution)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        total_loss = 0.0
        for voxel_batch, item_batch, cost_batch, label_batch in loader:
            optimizer.zero_grad()
            preds = model(voxel_batch, item_batch, cost_batch).squeeze(1)
            loss = loss_fn(preds, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * voxel_batch.size(0)
 
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
 
        # Has the loss improved by at least min_delta since the best we've seen?
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
 
        if epochs_without_improvement >= patience:
            print(f"Stopping early: loss hasn't improved by more than {min_delta} "
                  f"for {patience} epochs in a row.")
            break
 
    return model

# %%












# %%

######### INFERENCE / FILTERING (plugs into pack_item later) #########


def filter_bins(model, open_bins, item_dims, resolution=10, threshold=0.5):
    """
    Returns the subset of open_bins the NN scores as "good" candidates
    (score >= threshold) for the given item.
    """
    good_bins = []
    model.eval()
    with torch.no_grad():
        for bin_obj in open_bins:
            onehot = voxel_grid_to_onehot(bin_to_voxel_grid(bin_obj, resolution))
            voxel_tensor = torch.tensor(onehot, dtype=torch.float32).unsqueeze(0)
            item_tensor = torch.tensor(item_dims, dtype=torch.float32).unsqueeze(0)
            cost_tensor = torch.tensor([bin_obj.cost], dtype=torch.float32).unsqueeze(0)
            
            score = model(voxel_tensor, item_tensor, cost_tensor).item()
            if score >= threshold:
                good_bins.append(bin_obj)
    model.train()
    return good_bins

# %%

















# %%

############ MAIN - build data, train, save #################


if __name__ == "__main__":
    from online_generator import bin_types
    import json

    RESOLUTION = 10

    with open('bed-bpp_v1.json', 'r') as f:
        bed_bpp_data = json.load(f)
        
    sequences = []
    for order_id, order_data in bed_bpp_data.items():
        sequence = []
        items = order_data["item_sequence"]
        for item_key in sorted(items.keys(), key=int):
            item = items[item_key]
            sequence.append([item["length/mm"], item["width/mm"], item["height/mm"], item["weight/kg"]])
        sequences.append(sequence)

    # 80/20 Split
    
    train_size = int(len(sequences) * 0.8)
    train_sequences = sequences[:train_size]

    print("Building training data on 80% of the dataset...")
    # Removed max_sequences limit
    samples = build_training_data(train_sequences, bin_types, resolution=RESOLUTION)
    print(f"Generated {len(samples)} training samples")

    if samples:
        # Increased epochs to 50
        model = train_model(samples, resolution=RESOLUTION, epochs=50)
        torch.save(model.state_dict(), "bin_filter_nn.pt")
        print("Model saved to bin_filter_nn.pt")
    else:
        print("No training samples generated, check that sequences produce multi-bin scenarios.")


# %%