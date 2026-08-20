# %%
import time
import torch

import neural_net
import xgb_filter
import ems_heuristic
import ems2_heuristic
import extreme_points
from online_generator import bin_types

# %%






# %%

######### THE FUSION WRAPPER  ##########


def filtered_pack_item(pack_function, model, bin_types, open_bins,
                        item_l, item_w, item_h, resolution=10, threshold=0.5,
                        filter_kind="nn"):
    """
    Runs an ML bin filter, then hands only the surviving candidate bins to
    the given heuristic's pack_item function.

    pack_function - one of ems_heuristic.pack_item, ems2_heuristic.pack_item,
                    or extreme_points.pack_item (all share the same signature)
    model         - trained BinFilterNet (filter_kind="nn") or XGBClassifier
                    (filter_kind="xgb")
    open_bins     - the REAL list of currently open bins (mutated in place)
    filter_kind   - "nn" or "xgb"

    Returns the bin the item was placed in (same as pack_function normally does).
    """

    item_dims = (item_l, item_w, item_h)

    if open_bins:
        if filter_kind == "xgb":
            candidates = xgb_filter.filter_bins(
                model, open_bins, item_dims, threshold=threshold
            )
        else:
            candidates = neural_net.filter_bins(
                model, open_bins, item_dims, resolution=resolution, threshold=threshold
            )
    else:
        candidates = []

    # If the filter removed every bin (or there were none to begin with),
    # fall back to the full list so the heuristic can still place the item
    # or open a new bin as needed.
    if not candidates:
        candidates = open_bins

    result_bin = pack_function(bin_types, candidates, item_l, item_w, item_h)

    # If pack_function opened a brand new bin, it appended it to `candidates`.
    # When candidates is a separate filtered list (not open_bins itself),
    # that new bin never made it into the real open_bins list.
    if candidates is not open_bins and result_bin is not None and result_bin not in open_bins:
        open_bins.append(result_bin)

    return result_bin

# %%






# %%

####### RUNNING ONE SEQUENCE THROUGH ONE CONFIGURATION ######


def run_sequence(sequence, pack_function, model=None, resolution=10, threshold=0.5,
                 filter_kind="nn"):
    """
    Packs one full sequence using either the plain heuristic (model=None)
    or an ML-filtered version (model given; filter_kind "nn" or "xgb").

    Returns: (num_bins_used, total_cost, elapsed_seconds)
    """
    open_bins = []

    start = time.perf_counter()

    for item in sequence:
        item_l, item_w, item_h = item[0], item[1], item[2]

        if model is None:
            pack_function(bin_types, open_bins, item_l, item_w, item_h)
        else:
            filtered_pack_item(
                pack_function, model, bin_types, open_bins,
                item_l, item_w, item_h, resolution=resolution, threshold=threshold,
                filter_kind=filter_kind,
            )

    elapsed = time.perf_counter() - start

    total_cost = sum(b.cost for b in open_bins)
    num_bins_used = len(open_bins)

    return num_bins_used, total_cost, elapsed

# %%






# %%
######## COMPARISON ########


HEURISTICS = {
    "F53": ems_heuristic.pack_item,
    "F73": ems2_heuristic.pack_item,
    "EP": extreme_points.pack_item,
}


def run_comparison(sequences, model, resolution=10, threshold=0.5, filter_kind="nn"):
    """
    Runs every heuristic with and without the ML filter.

    results[heuristic_name]["baseline" | "with_filter"] = {
        "total_cost", "total_bins", "total_time"
    }
    """
    mode_key = "with_filter"

    results = {
        name: {
            "baseline": {"total_cost": 0, "total_bins": 0, "total_time": 0.0},
            mode_key: {"total_cost": 0, "total_bins": 0, "total_time": 0.0},
        }
        for name in HEURISTICS
    }

    for name, pack_function in HEURISTICS.items():
        for sequence in sequences:
            bins_used, cost, elapsed = run_sequence(sequence, pack_function, model=None)
            results[name]["baseline"]["total_cost"] += cost
            results[name]["baseline"]["total_bins"] += bins_used
            results[name]["baseline"]["total_time"] += elapsed

            bins_used, cost, elapsed = run_sequence(
                sequence, pack_function, model=model, resolution=resolution,
                threshold=threshold, filter_kind=filter_kind,
            )
            results[name][mode_key]["total_cost"] += cost
            results[name][mode_key]["total_bins"] += bins_used
            results[name][mode_key]["total_time"] += elapsed

    return results


def run_triple_comparison(sequences, nn_model, xgb_model, resolution=10, threshold=0.5):
    """
    Baseline vs NN filter vs XGB filter for each heuristic.
    """
    results = {
        name: {
            "baseline": {"total_cost": 0, "total_bins": 0, "total_time": 0.0},
            "with_nn": {"total_cost": 0, "total_bins": 0, "total_time": 0.0},
            "with_xgb": {"total_cost": 0, "total_bins": 0, "total_time": 0.0},
        }
        for name in HEURISTICS
    }

    for name, pack_function in HEURISTICS.items():
        for sequence in sequences:
            bins_used, cost, elapsed = run_sequence(sequence, pack_function, model=None)
            results[name]["baseline"]["total_cost"] += cost
            results[name]["baseline"]["total_bins"] += bins_used
            results[name]["baseline"]["total_time"] += elapsed

            bins_used, cost, elapsed = run_sequence(
                sequence, pack_function, model=nn_model, resolution=resolution,
                threshold=threshold, filter_kind="nn",
            )
            results[name]["with_nn"]["total_cost"] += cost
            results[name]["with_nn"]["total_bins"] += bins_used
            results[name]["with_nn"]["total_time"] += elapsed

            bins_used, cost, elapsed = run_sequence(
                sequence, pack_function, model=xgb_model, resolution=resolution,
                threshold=threshold, filter_kind="xgb",
            )
            results[name]["with_xgb"]["total_cost"] += cost
            results[name]["with_xgb"]["total_bins"] += bins_used
            results[name]["with_xgb"]["total_time"] += elapsed

    return results


def print_comparison(results, filter_label="NN"):
    mode_key = "with_filter" if "with_filter" in next(iter(results.values())) else f"with_{filter_label.lower()}"

    print("\n" + "=" * 70)
    print(f" HEURISTIC COMPARISON: BASELINE vs {filter_label}-FILTERED")
    print("=" * 70)

    for name, modes in results.items():
        base = modes["baseline"]
        filt = modes[mode_key]

        cost_change = filt["total_cost"] - base["total_cost"]
        time_change = filt["total_time"] - base["total_time"]

        print(f"\n{name}")
        print(f"  Baseline : cost={base['total_cost']:.0f}  bins={base['total_bins']}  time={base['total_time']:.3f}s")
        print(f"  With {filter_label:3s}: cost={filt['total_cost']:.0f}  bins={filt['total_bins']}  time={filt['total_time']:.3f}s")
        print(f"  Change   : cost={cost_change:+.0f}  time={time_change:+.3f}s")

    print("=" * 70)


def print_triple_comparison(results):
    print("\n" + "=" * 78)
    print(" HEURISTIC COMPARISON: BASELINE vs NN vs XGB")
    print("=" * 78)

    for name, modes in results.items():
        base = modes["baseline"]
        nn = modes["with_nn"]
        xgb = modes["with_xgb"]

        print(f"\n{name}")
        print(f"  Baseline : cost={base['total_cost']:.0f}  bins={base['total_bins']}  time={base['total_time']:.3f}s")
        print(f"  With NN  : cost={nn['total_cost']:.0f}  bins={nn['total_bins']}  time={nn['total_time']:.3f}s"
              f"  (dCost={nn['total_cost'] - base['total_cost']:+.0f})")
        print(f"  With XGB : cost={xgb['total_cost']:.0f}  bins={xgb['total_bins']}  time={xgb['total_time']:.3f}s"
              f"  (dCost={xgb['total_cost'] - base['total_cost']:+.0f})")

    print("=" * 78)


# %%






# %%
########### MAIN ###########


if __name__ == "__main__":
    import json

    RESOLUTION = 10
    THRESHOLD = 0.3  # Lowered threshold to reduce false negatives

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
    test_sequences = sequences[train_size:]

    nn_model = neural_net.BinFilterNet(resolution=RESOLUTION)
    nn_model.load_state_dict(torch.load("bin_filter_nn.pt", weights_only=True))
    nn_model.eval()

    try:
        xgb_model = xgb_filter.load_model("bin_filter_xgb.joblib")
    except FileNotFoundError:
        print("Training XGB filter on training set...")
        X, y = xgb_filter.build_training_data(train_sequences, bin_types)
        xgb_model = xgb_filter.train_model(X, y)
        xgb_filter.save_model(xgb_model, "bin_filter_xgb.joblib")

    # Evaluate strictly on test_sequences
    results = run_triple_comparison(
        test_sequences, nn_model, xgb_model, resolution=RESOLUTION, threshold=THRESHOLD
    )
    print_triple_comparison(results)

# %%
