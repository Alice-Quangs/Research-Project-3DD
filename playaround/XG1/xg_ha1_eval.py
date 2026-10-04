
import time

import xg_ha1
import ha1
from bin_types import bin_types

# %%
######### FUSION WRAPPER ##########

def filtered_pack_item(pack_function, model, bin_types, open_bins,
                        item_l, item_w, item_h, threshold=0.3):
    """
    Runs the XGB bin filter, then hands only the surviving candidate bins to
    the given heuristic's pack_item function.

    pack_function - for the F53 heuristic
    model         - trained XGBClassifier
    open_bins     - the REAL list of currently open bins (mutated in place)
    """

    item_dims = (item_l, item_w, item_h)

    if open_bins:
        candidates = xg_ha1.filter_bins(
            model, open_bins, item_dims, threshold=threshold
        )
    else:
        candidates = []

    # If the filter removed every bin, fall back to the full list
    if not candidates:
        candidates = open_bins

    result_bin = pack_function(bin_types, candidates, item_l, item_w, item_h)

    # Re-append if a new bin was created and candidate list was disconnected
    if candidates is not open_bins and result_bin is not None and result_bin not in open_bins:
        open_bins.append(result_bin)

    return result_bin

# %%
####### RUNNING ONE SEQUENCE THROUGH ONE CONFIGURATION ######

def run_sequence(sequence, pack_function, model=None, threshold=0.3):
    """
    Packs one full sequence using either the plain heuristic (model=None)
    or the XGB-filtered version for the F53 heuristic.
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
                item_l, item_w, item_h, threshold=threshold,
            )

    elapsed = time.perf_counter() - start

    total_cost = sum(b.cost for b in open_bins)
    num_bins_used = len(open_bins)

    total_bin_volume = sum(xg_ha1.bin_volume(b) for b in open_bins)
    total_used_volume = sum(xg_ha1.bin_used_volume(b) for b in open_bins)
    utilization = total_used_volume / total_bin_volume if total_bin_volume > 0 else 0.0
    residual_capacity = total_bin_volume - total_used_volume

    return num_bins_used, total_cost, elapsed, utilization, residual_capacity

# %%
######## COMPARISON ########

HEURISTICS = {
    "F53": ha1.pack_item,
}

def run_comparison(sequences, model, threshold=0.3):
    results = {
        name: {
            "baseline": {"total_cost": 0, "total_bins": 0, "total_time": 0.0,
                         "utilization_sum": 0.0, "total_residual_capacity": 0.0},
            "with_filter": {"total_cost": 0, "total_bins": 0, "total_time": 0.0,
                             "utilization_sum": 0.0, "total_residual_capacity": 0.0},
        }
        for name in HEURISTICS
    }

    n_sequences = len(sequences)

    for name, pack_function in HEURISTICS.items():
        for sequence in sequences:
            bins_used, cost, elapsed, util, residual = run_sequence(sequence, pack_function, model=None)
            results[name]["baseline"]["total_cost"] += cost
            results[name]["baseline"]["total_bins"] += bins_used
            results[name]["baseline"]["total_time"] += elapsed
            results[name]["baseline"]["utilization_sum"] += util
            results[name]["baseline"]["total_residual_capacity"] += residual

            bins_used, cost, elapsed, util, residual = run_sequence(
                sequence, pack_function, model=model, threshold=threshold,
            )
            results[name]["with_filter"]["total_cost"] += cost
            results[name]["with_filter"]["total_bins"] += bins_used
            results[name]["with_filter"]["total_time"] += elapsed
            results[name]["with_filter"]["utilization_sum"] += util
            results[name]["with_filter"]["total_residual_capacity"] += residual

    for name in results:
        for mode in ("baseline", "with_filter"):
            results[name][mode]["avg_utilization"] = (
                results[name][mode]["utilization_sum"] / n_sequences if n_sequences else 0.0
            )
            del results[name][mode]["utilization_sum"]

    return results

def print_comparison(results):
    print("\n" + "=" * 70)
    print(" HEURISTIC COMPARISON: BASELINE vs XGB-FILTERED")
    print("=" * 70)

    for name, modes in results.items():
        base = modes["baseline"]
        filt = modes["with_filter"]

        cost_change = filt["total_cost"] - base["total_cost"]
        time_change = filt["total_time"] - base["total_time"]

        base_residual_l = base["total_residual_capacity"] / 1e6
        filt_residual_l = filt["total_residual_capacity"] / 1e6

        print(f"\n{name}")
        print(f"  Baseline : cost={base['total_cost']:.0f}  bins={base['total_bins']}  "
              f"time={base['total_time']:.3f}s  util={base['avg_utilization']:.3f}  "
              f"residual={base_residual_l:,.1f} L")
        print(f"  With XGB : cost={filt['total_cost']:.0f}  bins={filt['total_bins']}  "
              f"time={filt['total_time']:.3f}s  util={filt['avg_utilization']:.3f}  "
              f"residual={filt_residual_l:,.1f} L")
        print(f"  Change   : cost={cost_change:+.0f}  time={time_change:+.3f}s")

    print("=" * 70)

# %%
######## SAMPLE-SIZE SWEEP ########

def run_sample_size_sweep(test_sequences, model, fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
                           threshold=0.3, seed=42):
    import random

    rng = random.Random(seed)
    n_total = len(test_sequences)
    sweep_results = []

    for frac in fractions:
        n = max(1, int(round(n_total * frac)))
        subset = rng.sample(test_sequences, n) if n < n_total else list(test_sequences)

        base_bins = base_cost = base_time = base_util_sum = base_residual = 0.0
        filt_bins = filt_cost = filt_time = filt_util_sum = filt_residual = 0.0

        for sequence in subset:
            bins_used, cost, elapsed, util, residual = run_sequence(
                sequence, ha1.pack_item, model=None
            )
            base_bins += bins_used
            base_cost += cost
            base_time += elapsed
            base_util_sum += util
            base_residual += residual

            bins_used, cost, elapsed, util, residual = run_sequence(
                sequence, ha1.pack_item, model=model, threshold=threshold
            )
            filt_bins += bins_used
            filt_cost += cost
            filt_time += elapsed
            filt_util_sum += util
            filt_residual += residual

        sweep_results.append({
            "fraction": frac,
            "n_sequences": n,
            "baseline": {
                "total_bins": base_bins, "total_cost": base_cost, "total_time": base_time,
                "avg_utilization": base_util_sum / n, "total_residual_capacity": base_residual,
            },
            "with_filter": {
                "total_bins": filt_bins, "total_cost": filt_cost, "total_time": filt_time,
                "avg_utilization": filt_util_sum / n, "total_residual_capacity": filt_residual,
            },
        })

    return sweep_results

def print_sample_size_sweep(sweep_results, label="F53"):
    print("\n" + "=" * 78)
    print(f" SAMPLE-SIZE SWEEP: {label} baseline vs XGB-filtered")
    print("=" * 78)
    print(f"{'n':>6} {'mode':<12} {'bins':>6} {'cost':>10} {'time(s)':>9} "
          f"{'util':>7} {'residual(L)':>14}")
    for row in sweep_results:
        n = row["n_sequences"]
        for mode_label, mode in (("baseline", row["baseline"]), ("with_filter", row["with_filter"])):
            residual_l = mode["total_residual_capacity"] / 1e6
            print(f"{n:>6} {mode_label:<12} {mode['total_bins']:>6.0f} {mode['total_cost']:>10.0f} "
                  f"{mode['total_time']:>9.3f} {mode['avg_utilization']:>7.3f} "
                  f"{residual_l:>14,.1f}")
    print("=" * 78)

def plot_sample_size_sweep(sweep_results, label="F53",
                            save_path="f53_sample_size_sweep.png", show=True):
    import matplotlib.pyplot as plt

    ns = [row["n_sequences"] for row in sweep_results]
    base = [row["baseline"] for row in sweep_results]
    filt = [row["with_filter"] for row in sweep_results]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def _plot(ax, key, ylabel, title, scale=1.0):
        ax.plot(ns, [b[key] * scale for b in base], marker="o", label="Baseline")
        ax.plot(ns, [f[key] * scale for f in filt], marker="o", label="With XGB")
        ax.set_xlabel("Number of test sequences")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    _plot(axes[0, 0], "avg_utilization", "Avg utilization", "Utilization vs sample size")
    _plot(axes[0, 1], "total_bins", "Total bins used", "Bins used vs sample size")
    _plot(axes[1, 0], "total_residual_capacity", "Total residual capacity (L)",
          "Residual capacity vs sample size", scale=1e-6)
    _plot(axes[1, 1], "total_time", "Time (s)", "Runtime vs sample size")

    fig.suptitle(f"{label}: Sample-size sweep")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Sample-size sweep plot saved to {save_path}")
    if show:
        plt.show()

    return fig

# %%
######## THRESHOLD SWEEP ########

def run_threshold_sweep(test_sequences, model, X_test, y_test,
                         thresholds=(0.1, 0.3, 0.5, 0.7, 0.9), sequences_subset=None):
    sequences = sequences_subset if sequences_subset is not None else test_sequences
    n = len(sequences)

    base_bins = base_cost = base_time = base_util_sum = base_residual = 0.0
    for sequence in sequences:
        bins_used, cost, elapsed, util, residual = run_sequence(sequence, ha1.pack_item, model=None)
        base_bins += bins_used
        base_cost += cost
        base_time += elapsed
        base_util_sum += util
        base_residual += residual

    baseline_packing = {
        "total_bins": base_bins, "total_cost": base_cost, "total_time": base_time,
        "avg_utilization": base_util_sum / n if n else 0.0,
        "total_residual_capacity": base_residual,
    }

    sweep_rows = []
    for t in thresholds:
        clf_metrics = xg_ha1.classification_metrics(model, X_test, y_test, threshold=t)

        filt_bins = filt_cost = filt_time = filt_util_sum = filt_residual = 0.0
        for sequence in sequences:
            bins_used, cost, elapsed, util, residual = run_sequence(
                sequence, ha1.pack_item, model=model, threshold=t
            )
            filt_bins += bins_used
            filt_cost += cost
            filt_time += elapsed
            filt_util_sum += util
            filt_residual += residual

        packing_metrics = {
            "total_bins": filt_bins, "total_cost": filt_cost, "total_time": filt_time,
            "avg_utilization": filt_util_sum / n if n else 0.0,
            "total_residual_capacity": filt_residual,
        }

        sweep_rows.append({"threshold": t, "classifier": clf_metrics, "packing": packing_metrics})

    return baseline_packing, sweep_rows


def print_threshold_sweep(baseline_packing, sweep_rows, label="F53"):
    print("\n" + "=" * 92)
    print(f" THRESHOLD SWEEP: {label}")
    print("=" * 92)

    base_residual_l = baseline_packing["total_residual_capacity"] / 1e6
    print(f"Baseline (no filter): bins={baseline_packing['total_bins']:.0f}  "
          f"cost={baseline_packing['total_cost']:.0f}  time={baseline_packing['total_time']:.3f}s  "
          f"util={baseline_packing['avg_utilization']:.3f}  residual={base_residual_l:,.1f} L")
    print("-" * 92)
    print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'f1':>7} {'fpr':>7} {'fnr':>7} "
          f"{'bins':>6} {'cost':>10} {'time(s)':>9} {'util':>7} {'residual(L)':>13}")
    for row in sweep_rows:
        t = row["threshold"]
        c = row["classifier"]
        p = row["packing"]
        residual_l = p["total_residual_capacity"] / 1e6
        print(f"{t:>5.2f} {c['precision']:>7.3f} {c['recall']:>7.3f} {c['f1']:>7.3f} "
              f"{c['fpr']:>7.3f} {c['fnr']:>7.3f} {p['total_bins']:>6.0f} {p['total_cost']:>10.0f} "
              f"{p['total_time']:>9.3f} {p['avg_utilization']:>7.3f} {residual_l:>13,.1f}")
    print("=" * 92)


def plot_threshold_sweep(baseline_packing, sweep_rows, label="F53",
                          save_path="f53_threshold_sweep.png", show=True):
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in sweep_rows]
    precision = [row["classifier"]["precision"] for row in sweep_rows]
    recall = [row["classifier"]["recall"] for row in sweep_rows]
    f1 = [row["classifier"]["f1"] for row in sweep_rows]
    fpr = [row["classifier"]["fpr"] for row in sweep_rows]
    fnr = [row["classifier"]["fnr"] for row in sweep_rows]
    bins = [row["packing"]["total_bins"] for row in sweep_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(thresholds, precision, marker="o", label="Precision")
    ax.plot(thresholds, recall, marker="o", label="Recall")
    ax.plot(thresholds, f1, marker="o", label="F1")
    ax.plot(thresholds, fpr, marker="o", label="FPR")
    ax.plot(thresholds, fnr, marker="o", label="FNR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Classifier metrics vs threshold")
    ax.legend()

    ax = axes[1]
    ax.plot(thresholds, bins, marker="o", label="With XGB (bins)")
    ax.axhline(baseline_packing["total_bins"], color="gray", linestyle="--", label="Baseline (bins)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total bins used")
    ax.set_title("Bins used vs threshold")
    ax.legend()

    fig.suptitle(f"{label}: Threshold sweep")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Threshold sweep plot saved to {save_path}")
    if show:
        plt.show()

    return fig

# %%
########### MAIN ###########

if __name__ == "__main__":
    import bed_loader

    THRESHOLD = 0.3  

    sequences = bed_loader.load_all_sequences("bed-bpp_v1.json")

    # 80/20 Split
    train_size = int(len(sequences) * 0.8)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    try:
        xgb_model = xg_ha1.load_model("bin_filter_xgb_ha1.joblib")
    except FileNotFoundError:
        print("Training XGB filter on training set...")
        X, y = xg_ha1.build_training_data(train_sequences, bin_types)
        xgb_model = xg_ha1.train_model(X, y)
        xg_ha1.save_model(xgb_model, "bin_filter_xgb_ha1.joblib")

    results = run_comparison(test_sequences, xgb_model, threshold=THRESHOLD)
    print_comparison(results)

    xg_ha1.plot_feature_importance(xgb_model, top_n=15, show=False)

    X_test, y_test = xg_ha1.build_training_data(test_sequences, bin_types)
    f53_metrics = xg_ha1.classification_metrics(xgb_model, X_test, y_test, threshold=THRESHOLD)
    xg_ha1.print_classification_metrics(f53_metrics, label="F53 (ha1) XGB filter")

    xg_ha1.plot_confusion_matrix(xgb_model, X_test, y_test, threshold=THRESHOLD,
                                  save_path="f53_confusion_matrix.png", show=False)
    xg_ha1.plot_roc_curve(xgb_model, X_test, y_test, save_path="f53_roc_curve.png", show=False)

    sweep = run_sample_size_sweep(test_sequences, xgb_model, threshold=THRESHOLD)
    print_sample_size_sweep(sweep, label="F53")
    plot_sample_size_sweep(sweep, label="F53", show=False)

    base_k, sweep_k = run_threshold_sweep(
        test_sequences, xgb_model, X_test, y_test,
        thresholds=(0.1, 0.3, 0.5, 0.7, 0.9),
        sequences_subset=test_sequences[:300]
    )
    print_threshold_sweep(base_k, sweep_k, label="F53")
    plot_threshold_sweep(base_k, sweep_k, label="F53", show=False)