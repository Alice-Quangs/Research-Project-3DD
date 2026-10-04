# %%
import time

import xg_ha2
import ha2
from bin_types import bin_types

# %%

######### FUSION WRAPPER ##########


def filtered_pack_item(pack_function, model, bin_types, open_bins,
                        item_l, item_w, item_h, threshold=0.5):
    """
    Runs the XGB bin filter, then hands only the surviving candidate bins to
    the given heuristic's pack_item function.

    pack_function - for the F73 heuristic
    model         - trained XGBClassifier
    open_bins     - the REAL list of currently open bins (mutated in place)

    Returns the bin the item was placed in (same as pack_function normally does).
    """

    item_dims = (item_l, item_w, item_h)

    if open_bins:
        candidates = xg_ha2.filter_bins(
            model, open_bins, item_dims, threshold=threshold
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

####### RUNNING ONE SEQUENCE THROUGH ONE CONFIGURATION ######


def run_sequence(sequence, pack_function, model=None, threshold=0.5):
    """
    Packs one full sequence using either the plain heuristic (model=None)
    or the XGB-filtered version for the F73 heuristic.

    Returns: (num_bins_used, total_cost, elapsed_seconds, utilization, residual_capacity)
    utilization        - total used volume across open bins / total bin volume (0-1)
    residual_capacity  - total bin volume - total used volume (leftover space, in
                          the same volume units as the item dimensions)
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

    total_bin_volume = sum(xg_ha2.bin_volume(b) for b in open_bins)
    total_used_volume = sum(xg_ha2.bin_used_volume(b) for b in open_bins)
    utilization = total_used_volume / total_bin_volume if total_bin_volume > 0 else 0.0
    residual_capacity = total_bin_volume - total_used_volume

    return num_bins_used, total_cost, elapsed, utilization, residual_capacity

# %%
######## COMPARISON ########


HEURISTICS = {
    "F73": ha2.pack_item,
}


def run_comparison(sequences, model, threshold=0.5):
    """
    Runs every heuristic with and without the XGB filter.

    results[heuristic_name]["baseline" | "with_filter"] = {
        "total_cost", "total_bins", "total_time",
        "avg_utilization", "total_residual_capacity"
    }
    avg_utilization is the mean per-sequence utilization (not volume-weighted),
    so each order contributes equally regardless of how many bins it used.
    """
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

    # convert running sums into per-sequence averages for utilization
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

        # item dims in the bed-bpp dataset are in mm, so bin_volume()/
        # residual capacity come out in mm^3 -> convert to litres for a
        # human-readable number (1 L = 1,000,000 mm^3).
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
######## SAMPLE-SIZE SWEEP (F73 / ha2 only, for now) ########


def run_sample_size_sweep(test_sequences, model, fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
                           threshold=0.5, seed=42):
    """
    Runs the F73 (ha2) heuristic, baseline vs XGB-filtered, on increasing
    fractions of the test sequences, so you can see how bins/cost/time/
    utilization/residual capacity behave as sample size grows.

    A random (seeded) subset is drawn from test_sequences for each fraction,
    rather than always taking the first N, to avoid any ordering bias in the
    dataset (e.g. if test_sequences happen to be grouped/ordered somehow).

    Returns a list of dicts, one per fraction, each with baseline and
    with_filter sub-dicts.
    """
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
                sequence, ha2.pack_item, model=None
            )
            base_bins += bins_used
            base_cost += cost
            base_time += elapsed
            base_util_sum += util
            base_residual += residual

            bins_used, cost, elapsed, util, residual = run_sequence(
                sequence, ha2.pack_item, model=model, threshold=threshold
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


def print_sample_size_sweep(sweep_results, label="F73"):
    print("\n" + "=" * 78)
    print(f" SAMPLE-SIZE SWEEP: {label} baseline vs XGB-filtered")
    print("=" * 78)
    print(f"{'n':>6} {'mode':<12} {'bins':>6} {'cost':>10} {'time(s)':>9} "
          f"{'util':>7} {'residual(L)':>14}")
    for row in sweep_results:
        n = row["n_sequences"]
        for mode_label, mode in (("baseline", row["baseline"]), ("with_filter", row["with_filter"])):
            residual_l = mode["total_residual_capacity"] / 1e6  # mm^3 -> litres
            print(f"{n:>6} {mode_label:<12} {mode['total_bins']:>6.0f} {mode['total_cost']:>10.0f} "
                  f"{mode['total_time']:>9.3f} {mode['avg_utilization']:>7.3f} "
                  f"{residual_l:>14,.1f}")
    print("=" * 78)


def plot_sample_size_sweep(sweep_results, label="F73",
                            save_path="f73_sample_size_sweep.png", show=True):
    """
    Utilization stays a line (its range is too narrow for bars to read).
    Bins used, bin cost, and residual capacity are grouped bars,
    baseline vs XGB-filtered, at each sample size.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ns = [row["n_sequences"] for row in sweep_results]
    base = [row["baseline"] for row in sweep_results]
    filt = [row["with_filter"] for row in sweep_results]

    x = np.arange(len(ns))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # --- Utilization: keep as a line, its range is too narrow for bars ---
    ax = axes[0, 0]
    ax.plot(ns, [b["avg_utilization"] for b in base], marker="o", label="Baseline")
    ax.plot(ns, [f["avg_utilization"] for f in filt], marker="o", label="With XGB")
    ax.set_xlabel("Number of test sequences")
    ax.set_ylabel("Avg utilization")
    ax.set_title("Utilization vs sample size")
    ax.legend()

    def _bar(ax, key, ylabel, title, scale=1.0):
        base_vals = [b[key] * scale for b in base]
        filt_vals = [f[key] * scale for f in filt]
        ax.bar(x - width / 2, base_vals, width, label="Baseline")
        ax.bar(x + width / 2, filt_vals, width, label="With XGB")
        ax.set_xticks(x)
        ax.set_xticklabels(ns)
        ax.set_xlabel("Number of test sequences")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    _bar(axes[0, 1], "total_bins", "Total bins used", "Bins used vs sample size")
    # residual capacity is stored in mm^3 (bed-bpp item dims are in mm) -> litres for readability
    _bar(axes[1, 0], "total_residual_capacity", "Total residual capacity (L)",
         "Residual capacity vs sample size", scale=1e-6)
    # swapped out runtime (not performing as expected) for bin cost
    _bar(axes[1, 1], "total_cost", "Total bin cost", "Bin cost vs sample size")

    fig.suptitle(f"{label}: Sample-size sweep")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Sample-size sweep plot saved to {save_path}")
    if show:
        plt.show()

    return fig


# %%
######## THRESHOLD SWEEP (F73 / ha2 only, for now) ########


def run_threshold_sweep(test_sequences, model, X_test, y_test,
                         thresholds=(0.1, 0.3, 0.5, 0.7, 0.9), sequences_subset=None):
    """
    Sweeps the XGB filter's decision threshold and reports, at each value:
      - classifier metrics (precision, recall, F1, FPR, FNR) computed once
        on (X_test, y_test) at that threshold.
      - downstream packing metrics (bins, cost, time, utilization, residual
        capacity) from actually running F73 filtered by the model at that
        threshold across test_sequences (or sequences_subset if given, to
        iterate faster - each threshold requires a full filtered pass, which
        can be slow; see run_sample_size_sweep timings for reference).

    Also runs the unfiltered baseline once (threshold-independent) so the
    packing metrics can be compared against it.

    Returns (baseline_packing, sweep_rows) where sweep_rows is a list of
    dicts, one per threshold, each holding 'threshold', 'classifier', and
    'packing' sub-dicts.
    """
    sequences = sequences_subset if sequences_subset is not None else test_sequences
    n = len(sequences)

    # Baseline packing run (threshold-independent), once.
    base_bins = base_cost = base_time = base_util_sum = base_residual = 0.0
    for sequence in sequences:
        bins_used, cost, elapsed, util, residual = run_sequence(sequence, ha2.pack_item, model=None)
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
        clf_metrics = xg_ha2.classification_metrics(model, X_test, y_test, threshold=t)

        filt_bins = filt_cost = filt_time = filt_util_sum = filt_residual = 0.0
        for sequence in sequences:
            bins_used, cost, elapsed, util, residual = run_sequence(
                sequence, ha2.pack_item, model=model, threshold=t
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


def print_threshold_sweep(baseline_packing, sweep_rows, label="F73"):
    print("\n" + "=" * 92)
    print(f" THRESHOLD SWEEP: {label}")
    print("=" * 92)

    base_residual_l = baseline_packing["total_residual_capacity"] / 1e6  # mm^3 -> litres
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


def plot_threshold_sweep(baseline_packing, sweep_rows, label="F73",
                          save_path="f73_threshold_sweep.png", show=True):
    """
    Two-panel figure: left shows classifier metrics (precision/recall/F1/
    FPR/FNR) vs threshold; right shows total bins used vs threshold, with
    the unfiltered baseline drawn as a horizontal reference line.
    """
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

    THRESHOLD = 0.3  # Same starting point as F53; revisit once F53's sweep results are in

    sequences = bed_loader.load_all_sequences("bed-bpp_v1.json")

    # 80/20 Split
    train_size = int(len(sequences) * 0.8)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    try:
        xgb_model = xg_ha2.load_model("bin_filter_xgb_ha2.joblib")
    except FileNotFoundError:
        print("Training XGB filter on training set...")
        X, y = xg_ha2.build_training_data(train_sequences, bin_types)
        xgb_model = xg_ha2.train_model(X, y)
        xg_ha2.save_model(xgb_model, "bin_filter_xgb_ha2.joblib")

    # Evaluate strictly on test_sequences
    results = run_comparison(test_sequences, xgb_model, threshold=THRESHOLD)
    print_comparison(results)

    xg_ha2.plot_feature_importance(xgb_model, top_n=15, show=False)

    # --- Classifier-quality metrics for F73 (ha2), on held-out test sequences ---
    # build_training_data replays sequences with ha2.pack_item, so this matches
    # the F73 model's own labelling policy exactly.
    X_test, y_test = xg_ha2.build_training_data(test_sequences, bin_types)

    f73_metrics = xg_ha2.classification_metrics(xgb_model, X_test, y_test, threshold=THRESHOLD)
    xg_ha2.print_classification_metrics(f73_metrics, label="F73 (ha2) XGB filter")

    xg_ha2.plot_confusion_matrix(xgb_model, X_test, y_test, threshold=THRESHOLD,
                                  save_path="f73_confusion_matrix.png", show=False)
    xg_ha2.plot_roc_curve(xgb_model, X_test, y_test, save_path="f73_roc_curve.png", show=False)

    # --- Sample-size sweep for F73 (ha2) ---
    sweep = run_sample_size_sweep(test_sequences, xgb_model, threshold=THRESHOLD)
    print_sample_size_sweep(sweep, label="F73")
    plot_sample_size_sweep(sweep, label="F73", show=False)

    # --- Threshold sweep for F73 (ha2) ---
    # NOTE: this runs a full filtered pass over test_sequences for each
    # threshold, so it's slow. Pass a smaller sequences_subset=test_sequences[:N]
    # below if you want to iterate faster first.
    thr_baseline, thr_sweep = run_threshold_sweep(
        test_sequences, xgb_model, X_test, y_test,
        thresholds=(0.1, 0.3, 0.5, 0.7, 0.9),
    )
    print_threshold_sweep(thr_baseline, thr_sweep, label="F73")
    plot_threshold_sweep(thr_baseline, thr_sweep, label="F73", show=False)

# %%