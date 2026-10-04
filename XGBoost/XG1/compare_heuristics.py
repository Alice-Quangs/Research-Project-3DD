# compare_heuristics.py
#
# Compares F53, F73, and Extreme Points (EP) against each other directly —
# same test sequences, same bin types — for both plain heuristic packing
# and each heuristic's own trained XGB filter. Produces one bar chart per
# metric (bins / cost / utilization), each with a baseline and an
# XGB-filtered bar per heuristic, so you can see which heuristic wins,
# and whether ML filtering helps each one.

import time

import bed_loader
import ha1
import ha2
import ha3
import xg_ha1
import xg_ha2
import xg_ha3
from bin_types import bin_types

# name -> (pack_function, xg_module, model_path)
HEURISTICS = {
    "F53": (ha1.pack_item, xg_ha1, "bin_filter_xgb_ha1.joblib"),
    "F73": (ha2.pack_item, xg_ha2, "bin_filter_xgb_ha2.joblib"),
    "EP":  (ha3.pack_item, xg_ha3, "bin_filter_xgb_ha3.joblib"),
}

THRESHOLD = 0.3  # same threshold used in the individual xg_ha*_eval.py scripts


def filtered_pack_item(pack_function, xg_module, model, open_bins, item_l, item_w, item_h, threshold):
    """Generic version of each xg_ha*_eval.py's filtered_pack_item, parameterised
    over which heuristic + which trained filter module to use."""
    item_dims = (item_l, item_w, item_h)
    candidates = xg_module.filter_bins(model, open_bins, item_dims, threshold=threshold) if open_bins else []
    if not candidates:
        candidates = open_bins

    result_bin = pack_function(bin_types, candidates, item_l, item_w, item_h)

    if candidates is not open_bins and result_bin is not None and result_bin not in open_bins:
        open_bins.append(result_bin)

    return result_bin


def run_sequence(sequence, pack_function, xg_module=None, model=None, threshold=THRESHOLD):
    """Packs one sequence, baseline (model=None) or XGB-filtered.
    Returns (num_bins_used, total_cost, elapsed_seconds, utilization)."""
    open_bins = []
    start = time.perf_counter()

    for item in sequence:
        item_l, item_w, item_h = item[0], item[1], item[2]
        if model is None:
            pack_function(bin_types, open_bins, item_l, item_w, item_h)
        else:
            filtered_pack_item(pack_function, xg_module, model, open_bins,
                                item_l, item_w, item_h, threshold)

    elapsed = time.perf_counter() - start

    total_cost = sum(b.cost for b in open_bins)
    num_bins_used = len(open_bins)
    total_bin_volume = sum(b.length * b.width * b.height for b in open_bins)
    total_used_volume = sum(
        sum((x2 - x1) * (y2 - y1) * (z2 - z1) for (x1, y1, z1, x2, y2, z2) in b.placed_items)
        for b in open_bins
    )
    utilization = total_used_volume / total_bin_volume if total_bin_volume > 0 else 0.0

    return num_bins_used, total_cost, elapsed, utilization


def run_heuristic_comparison(sequences):
    """
    results[name]["baseline" | "with_filter"] = {total_bins, total_cost, total_time, avg_utilization}
    """
    results = {
        name: {
            "baseline": {"total_bins": 0, "total_cost": 0, "total_time": 0.0, "utilization_sum": 0.0},
            "with_filter": {"total_bins": 0, "total_cost": 0, "total_time": 0.0, "utilization_sum": 0.0},
        }
        for name in HEURISTICS
    }
    n = len(sequences)

    for name, (pack_function, xg_module, model_path) in HEURISTICS.items():
        model = xg_module.load_model(model_path)

        for sequence in sequences:
            bins_used, cost, elapsed, util = run_sequence(sequence, pack_function)
            results[name]["baseline"]["total_bins"] += bins_used
            results[name]["baseline"]["total_cost"] += cost
            results[name]["baseline"]["total_time"] += elapsed
            results[name]["baseline"]["utilization_sum"] += util

            bins_used, cost, elapsed, util = run_sequence(sequence, pack_function, xg_module, model, THRESHOLD)
            results[name]["with_filter"]["total_bins"] += bins_used
            results[name]["with_filter"]["total_cost"] += cost
            results[name]["with_filter"]["total_time"] += elapsed
            results[name]["with_filter"]["utilization_sum"] += util

    for name in results:
        for mode in ("baseline", "with_filter"):
            results[name][mode]["avg_utilization"] = (
                results[name][mode]["utilization_sum"] / n if n else 0.0
            )
            del results[name][mode]["utilization_sum"]

    return results


def print_heuristic_comparison(results):
    print("\n" + "=" * 78)
    print(" HEURISTIC COMPARISON: F53 vs F73 vs EP, baseline vs XGB-filtered")
    print("=" * 78)
    for name, modes in results.items():
        for mode_label, r in modes.items():
            print(f"{name:>4} {mode_label:<12}:  bins={r['total_bins']:>7,.0f}   "
                  f"cost={r['total_cost']:>9,.0f}   time={r['total_time']:>8.2f}s   "
                  f"util={r['avg_utilization']:.3f}")
    print("=" * 78)


def plot_heuristic_comparison(results, save_path="heuristic_comparison.png", show=True):
    """Grouped bars: one group per heuristic, baseline vs XGB-filtered side by side,
    one subplot per metric (bins / cost / utilization)."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def _bar(ax, key, ylabel, title):
        base_vals = [results[n]["baseline"][key] for n in names]
        filt_vals = [results[n]["with_filter"][key] for n in names]
        ax.bar(x - width / 2, base_vals, width, label="Baseline")
        ax.bar(x + width / 2, filt_vals, width, label="With XGB")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    _bar(axes[0], "total_bins", "Bins", "Total Bins Used")
    _bar(axes[1], "total_cost", "Cost", "Total Bin Cost")
    _bar(axes[2], "avg_utilization", "Utilization", "Avg Utilization")
    axes[2].set_ylim(0, 1)

    fig.suptitle("Heuristic Comparison: F53 vs F73 vs Extreme Points")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Heuristic comparison plot saved to {save_path}")
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    sequences = bed_loader.load_all_sequences("bed-bpp_v1.json")

    # Same 80/20 split convention as xg_ha1_eval.py / xg_ha2_eval.py / xg_ha3_eval.py,
    # so this compares on the same held-out test sequences they each use.
    train_size = int(len(sequences) * 0.8)
    test_sequences = sequences[train_size:]

    results = run_heuristic_comparison(test_sequences)
    print_heuristic_comparison(results)
    plot_heuristic_comparison(results, show=False)
