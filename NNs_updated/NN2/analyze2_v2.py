# analyze2_v2.py
#
# Reads all the .csv files ha2_nn_v2.py saved into the "Results" folder
# and turns them into plots, saved into a "Visuals" folder (created
# automatically). Does NOT do any packing or training itself. If the
# Results files don't exist yet, it runs ha2_nn_v2.py first.
#
# Plots produced:
#   1. Hyperparameter tuning plot   (accuracy + kappa, per size/decay)
#   2. Variable importance          (which features the NN relies on)
#   3. ROC curve + threshold choice (how we picked the decision threshold)
#   4. Confusion matrix             (TP / FP / TN / FN at that threshold)
#   5. Bucketed comparison          (baseline vs NN-filtered, by order size)
#   6. Threshold sweep              (classifier metrics + bins used)
#   7. NEW - Time breakdown         (NN inference vs geometry vs other,
#                                    by order-size bucket, with the
#                                    baseline's total time as a reference)
#   8. NEW - Inference benchmark    (sklearn predict_proba vs numpy
#                                    forward pass, at several batch sizes)
#   9. NEW - Filtering funnel       (bins screened -> rejected by NN ->
#                                    approved but failed geometry, plus
#                                    how many items skipped the NN
#                                    entirely via the <=1-open-bin rule)

import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no interactive window - avoids Qt-related save issues
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FOLDER = "Results"
VISUALS_FOLDER = "Visuals"

BUCKET_EDGES = [0, 20, 30, 40, 50, 60, 80, float("inf")]
BUCKET_LABELS = ["1-20", "21-30", "31-40", "41-50", "51-60", "61-80", "81+"]

BASELINE_COLOR = "#4C72B0"    # blue
NN_FILTER_COLOR = "#DD8452"   # orange
NN_TIME_COLOR = "#DD8452"     # orange
GEOM_TIME_COLOR = "#55A868"   # green
OTHER_TIME_COLOR = "#8C8C8C"  # gray


def save_figure(fig, path, max_attempts=5, delay=0.5):
    """
    Windows + OneDrive can transiently lock a freshly-created file (mid-
    sync, antivirus scan, etc.), which surfaces as a confusing
    'OSError: [Errno 22] Invalid argument' from PIL/matplotlib even though
    the path itself is fine. Retrying after a short pause almost always
    succeeds, so we do that instead of letting the whole run crash.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            return
        except OSError as e:
            last_error = e
            print(f"  (save attempt {attempt} failed - {e}; retrying...)")
            time.sleep(delay)
    raise last_error


# ============================================================
#  MAKING SURE THE RESULTS FILES ACTUALLY EXIST
# ============================================================

REQUIRED_RESULTS_FILES = [
    "ha2_grid_search_results.csv",
    "ha2_variable_importance.csv",
    "ha2_roc_curve.csv",
    "ha2_threshold_sweep.csv",
    "ha2_confusion_matrix.csv",
    "ha2_baseline_per_order.csv",
    "ha2_nn_filtered_per_order.csv",
    "ha2_nn_filtering_stats.csv",
    "ha2_nn_threshold_sweep.csv",
    "ha2_nn_speed_benchmark.csv",
]


def ensure_results_exist():
    missing = [
        name for name in REQUIRED_RESULTS_FILES
        if not os.path.exists(os.path.join(RESULTS_FOLDER, name))
    ]

    if missing:
        print("Some Results files are missing:")
        for name in missing:
            print(f"  - {name}")
        print("Running ha2_nn_v2.py first to generate them (this trains the NN and")
        print("runs the heuristic across the whole dataset, so it will take a while)...\n")

        import ha2_nn_v2
        ha2_nn_v2.main()
    else:
        print("All required Results files found - skipping straight to plotting.")


def load_results():
    def read(name):
        return pd.read_csv(os.path.join(RESULTS_FOLDER, name))

    return {
        "grid": read("ha2_grid_search_results.csv"),
        "importance": read("ha2_variable_importance.csv"),
        "roc": read("ha2_roc_curve.csv"),
        "sweep": read("ha2_threshold_sweep.csv"),
        "confusion": read("ha2_confusion_matrix.csv"),
        "baseline": read("ha2_baseline_per_order.csv"),
        "nn_filtered": read("ha2_nn_filtered_per_order.csv"),
        "filtering_stats": read("ha2_nn_filtering_stats.csv"),
        "nn_sweep": read("ha2_nn_threshold_sweep.csv"),
        "benchmark": read("ha2_nn_speed_benchmark.csv"),
    }


# ============================================================
#  PLOT 1: HYPERPARAMETER TUNING (ACCURACY + KAPPA)
# ============================================================

def plot_hyperparameter_tuning(grid_df):
    print("Plotting hyperparameter tuning results...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for decay_value in sorted(grid_df["decay"].unique()):
        subset = grid_df[grid_df["decay"] == decay_value].sort_values("hidden_layer_size")
        axes[0].plot(subset["hidden_layer_size"], subset["cv_accuracy"], marker="o", label=f"decay={decay_value}")
        axes[1].plot(subset["hidden_layer_size"], subset["cv_kappa"], marker="o", label=f"decay={decay_value}")

    axes[0].set_title("Cross-Validated Accuracy")
    axes[0].set_xlabel("Hidden layer size")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(title="Decay", fontsize=8)

    axes[1].set_title("Cross-Validated Kappa")
    axes[1].set_xlabel("Hidden layer size")
    axes[1].set_ylabel("Kappa")
    axes[1].legend(title="Decay", fontsize=8)

    fig.suptitle("HA2 NN Hyperparameter Tuning", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_hyperparameter_tuning.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 2: VARIABLE IMPORTANCE
# ============================================================

def plot_variable_importance(importance_df):
    print("Plotting variable importance...")

    df = importance_df.sort_values("importance_scaled", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["importance_scaled"], color=BASELINE_COLOR)
    ax.set_xlabel("Importance (scaled 0-100)")
    ax.set_title("HA2 NN Variable Importance")
    fig.tight_layout()

    save_path = os.path.join(VISUALS_FOLDER, "ha2_variable_importance.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 3: ROC CURVE + THRESHOLD SELECTION
# ============================================================

def plot_roc_and_threshold(roc_df, sweep_df, confusion_df):
    print("Plotting ROC curve and threshold selection...")

    chosen_threshold = confusion_df["chosen_threshold"].iloc[0]
    roc_auc = roc_df["roc_auc"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(roc_df["fpr"], roc_df["tpr"], color=BASELINE_COLOR, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

    chosen_row = sweep_df.iloc[(sweep_df["threshold"] - chosen_threshold).abs().idxmin()]
    axes[0].scatter([chosen_row["fpr"]], [chosen_row["tpr"]], color=NN_FILTER_COLOR, zorder=5,
                     label=f"Chosen threshold = {chosen_threshold:.2f}")

    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)

    axes[1].plot(sweep_df["threshold"], sweep_df["youden_j"], color=BASELINE_COLOR, label="Youden's J (TPR - FPR)")
    axes[1].axvline(chosen_threshold, color=NN_FILTER_COLOR, linestyle="--",
                     label=f"Chosen threshold = {chosen_threshold:.2f}")
    axes[1].set_title("Threshold Selection")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Youden's J")
    axes[1].legend(fontsize=8)

    fig.suptitle("HA2 NN: ROC Curve and Threshold Choice", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_roc_and_threshold.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 4: CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(confusion_df):
    print("Plotting confusion matrix...")

    row = confusion_df.iloc[0]
    matrix = np.array([
        [row["true_negative"], row["false_positive"]],
        [row["false_negative"], row["true_positive"]],
    ])
    labels = ["No Fit", "Fit"]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold = {row['chosen_threshold']:.2f})")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    footnote = (f"Precision={row['precision']:.3f}  Recall={row['recall']:.3f}  "
                f"F1={row['f1_score']:.3f}  Accuracy={row['accuracy']:.3f}  Kappa={row['kappa']:.3f}")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=9)

    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_confusion_matrix.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 5: BASELINE vs NN-FILTERED, BY ORDER SCALE BUCKET
# ============================================================

def bucket_and_summarize(df, label):
    df = df.copy()
    df["scale_bucket"] = pd.cut(df["item_count"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True)

    summary = df.groupby("scale_bucket", observed=True).agg(
        n_orders=("order_id", "count"),
        avg_bins_per_order=("bins_used", "mean"),
        avg_utilization_pct=("utilization_pct", "mean"),
        avg_cost=("cost", "mean"),
        avg_time_seconds=("time_seconds", "mean"),
        total_residual_capacity=("residual_capacity", "sum"),
    ).reset_index()
    summary["total_residual_capacity_l"] = summary["total_residual_capacity"] / 1e6
    summary["source"] = label
    return summary


def plot_bucket_comparison(baseline_df, nn_df):
    print("Plotting bucketed baseline vs NN-filtered comparison...")

    baseline_summary = bucket_and_summarize(baseline_df, "Baseline")
    nn_summary = bucket_and_summarize(nn_df, "NN filter")
    combined = pd.concat([baseline_summary, nn_summary], ignore_index=True)

    print("\n" + "=" * 80)
    print(" HA2 PERFORMANCE BY ORDER SCALE (BASELINE vs NN FILTER)")
    print("=" * 80)
    print(combined.to_string(index=False))

    pivot_bins = combined.pivot(index="scale_bucket", columns="source", values="avg_bins_per_order")
    pivot_util = combined.pivot(index="scale_bucket", columns="source", values="avg_utilization_pct")
    pivot_cost = combined.pivot(index="scale_bucket", columns="source", values="avg_cost")
    pivot_residual = combined.pivot(index="scale_bucket", columns="source", values="total_residual_capacity_l")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = [BASELINE_COLOR, NN_FILTER_COLOR]

    pivot_bins.plot(kind="bar", ax=axes[0, 0], color=colors)
    axes[0, 0].set_title("Avg Bins Used per Order")
    axes[0, 0].set_xlabel("Order size (items)")
    axes[0, 0].set_ylabel("Avg bins used")
    axes[0, 0].tick_params(axis="x", rotation=0)

    pivot_util.plot(kind="bar", ax=axes[0, 1], color=colors)
    axes[0, 1].set_title("Avg Space Utilization")
    axes[0, 1].set_xlabel("Order size (items)")
    axes[0, 1].set_ylabel("Utilization (%)")
    axes[0, 1].tick_params(axis="x", rotation=0)

    pivot_cost.plot(kind="bar", ax=axes[1, 0], color=colors)
    axes[1, 0].set_title("Avg Bin Cost per Order")
    axes[1, 0].set_xlabel("Order size (items)")
    axes[1, 0].set_ylabel("Avg cost")
    axes[1, 0].tick_params(axis="x", rotation=0)

    pivot_residual.plot(kind="bar", ax=axes[1, 1], color=colors)
    axes[1, 1].set_title("Total Residual Capacity")
    axes[1, 1].set_xlabel("Order size (items)")
    axes[1, 1].set_ylabel("Residual capacity (L)")
    axes[1, 1].tick_params(axis="x", rotation=0)

    fig.suptitle("HA2: Baseline vs NN Filter, by Order Scale", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_scale_comparison.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"\n  Saved: {save_path}")

    print("\n" + "=" * 80)
    print(" TIME SAVINGS SUMMARY (NN filter vs Baseline)")
    print("=" * 80)
    total_baseline_time = baseline_df["time_seconds"].sum()
    total_nn_time = nn_df["time_seconds"].sum()
    total_savings = ((total_baseline_time - total_nn_time) / total_baseline_time * 100) if total_baseline_time > 0 else 0
    print(f"  OVERALL: Baseline={total_baseline_time:.2f}s, NN-filtered={total_nn_time:.2f}s, "
          f"Savings={total_savings:+.1f}%\n")

    for bucket in BUCKET_LABELS:
        base_row = baseline_summary[baseline_summary["scale_bucket"] == bucket]
        nn_row = nn_summary[nn_summary["scale_bucket"] == bucket]
        if base_row.empty or nn_row.empty:
            continue
        baseline_time = base_row["avg_time_seconds"].values[0]
        nn_time = nn_row["avg_time_seconds"].values[0]
        n_orders = base_row["n_orders"].values[0]
        savings = ((baseline_time - nn_time) / baseline_time * 100) if baseline_time > 0 else 0
        print(f"  {bucket:>8} items ({n_orders:>5} orders): "
              f"Baseline={baseline_time:.4f}s, NN={nn_time:.4f}s, Savings={savings:+.1f}%")

    return total_baseline_time, total_nn_time


# ============================================================
#  PLOT 5b: AVERAGE TIME vs SEQUENCE LENGTH (BASELINE vs NN+HEURISTIC)
# ============================================================
#
# Line graph: x-axis is order size (sequence length), y-axis is average
# total time per order, one line for the plain heuristic and one for the
# NN-filtered (fast) heuristic. Uses the same scale buckets as the rest
# of the pipeline for consistency, but a finer per-length version is also
# produced underneath it for a less "steppy" picture.

def plot_time_vs_length(baseline_df, nn_df):
    print("Plotting average time vs sequence length (baseline vs NN+heuristic)...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel 1: bucketed (matches the rest of the pipeline's buckets) ---
    base_bucketed = baseline_df.copy()
    base_bucketed["scale_bucket"] = pd.cut(base_bucketed["item_count"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True)
    base_avg = base_bucketed.groupby("scale_bucket", observed=True)["time_seconds"].mean().reindex(BUCKET_LABELS)

    nn_bucketed = nn_df.copy()
    nn_bucketed["scale_bucket"] = pd.cut(nn_bucketed["item_count"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True)
    nn_avg = nn_bucketed.groupby("scale_bucket", observed=True)["time_seconds"].mean().reindex(BUCKET_LABELS)

    x = np.arange(len(BUCKET_LABELS))
    axes[0].plot(x, base_avg.values, marker="o", color=BASELINE_COLOR, label="Heuristic alone")
    axes[0].plot(x, nn_avg.values, marker="o", color=NN_FILTER_COLOR, label="NN + Heuristic")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BUCKET_LABELS)
    axes[0].set_xlabel("Order size (items)")
    axes[0].set_ylabel("Avg time per order (s)")
    axes[0].set_title("By order-size bucket")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Panel 2: finer granularity - average time per exact item count ---
    base_by_len = baseline_df.groupby("item_count")["time_seconds"].mean()
    nn_by_len = nn_df.groupby("item_count")["time_seconds"].mean()

    axes[1].plot(base_by_len.index, base_by_len.values, color=BASELINE_COLOR, alpha=0.8, label="Heuristic alone")
    axes[1].plot(nn_by_len.index, nn_by_len.values, color=NN_FILTER_COLOR, alpha=0.8, label="NN + Heuristic")
    axes[1].set_xlabel("Sequence length (items)")
    axes[1].set_ylabel("Avg time per order (s)")
    axes[1].set_title("By exact sequence length")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("HA2: Average Time vs Sequence Length (Heuristic Alone vs NN + Heuristic)", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_time_vs_length.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 6: THRESHOLD SWEEP (CLASSIFIER METRICS + BINS USED)
# ============================================================

def plot_threshold_sweep(sweep_df):
    print("Plotting threshold sweep...")

    thresholds = sweep_df["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(thresholds, sweep_df["precision"], marker="o", label="Precision")
    ax.plot(thresholds, sweep_df["recall"], marker="o", label="Recall")
    ax.plot(thresholds, sweep_df["f1"], marker="o", label="F1")
    ax.plot(thresholds, sweep_df["fpr"], marker="o", label="FPR")
    ax.plot(thresholds, sweep_df["fnr"], marker="o", label="FNR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Classifier metrics vs threshold")
    ax.legend()

    ax = axes[1]
    ax.plot(thresholds, sweep_df["bins_used"], marker="o", label="With NN (bins)")
    ax.axhline(sweep_df["baseline_bins"].iloc[0], color="gray", linestyle="--", label="Baseline (bins)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total bins used")
    ax.set_title("Bins used vs threshold")
    ax.legend()

    fig.suptitle("HA2 NN: Threshold sweep", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_threshold_sweep.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 7 (NEW): TIME BREAKDOWN BY ORDER-SIZE BUCKET
# ============================================================
#
# Shows where NN-filtered time actually goes (NN inference / geometry /
# other) per order-size bucket, with the baseline's average total time
# drawn as a reference line so you can see the fast NN path is genuinely
# cheaper, not just shifting the cost around.

def plot_time_breakdown(nn_df, baseline_df):
    print("Plotting NN time breakdown by order-size bucket...")

    df = nn_df.copy()
    df["scale_bucket"] = pd.cut(df["item_count"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True)
    breakdown = df.groupby("scale_bucket", observed=True).agg(
        avg_nn_time=("nn_inference_time_seconds", "mean"),
        avg_geom_time=("geometry_time_seconds", "mean"),
        avg_other_time=("other_time_seconds", "mean"),
    ).reindex(BUCKET_LABELS)

    base_df = baseline_df.copy()
    base_df["scale_bucket"] = pd.cut(base_df["item_count"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True)
    baseline_avg = base_df.groupby("scale_bucket", observed=True)["time_seconds"].mean().reindex(BUCKET_LABELS)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(BUCKET_LABELS))

    ax.bar(x, breakdown["avg_nn_time"], label="NN inference time", color=NN_TIME_COLOR)
    ax.bar(x, breakdown["avg_geom_time"], bottom=breakdown["avg_nn_time"], label="Geometry check time", color=GEOM_TIME_COLOR)
    ax.bar(x, breakdown["avg_other_time"],
           bottom=breakdown["avg_nn_time"] + breakdown["avg_geom_time"],
           label="Other (loop/bin management)", color=OTHER_TIME_COLOR)

    ax.plot(x, baseline_avg.values, color=BASELINE_COLOR, marker="o", linestyle="--",
            linewidth=2, label="Baseline avg total time")

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Order size (items)")
    ax.set_ylabel("Avg time per order (s)")
    ax.set_title("HA2 NN-Filtered: Where the Time Goes, by Order Scale")
    ax.legend()

    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_nn_time_breakdown.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 8 (NEW): INFERENCE BENCHMARK (sklearn vs numpy)
# ============================================================

def plot_inference_benchmark(bench_df):
    print("Plotting sklearn vs numpy inference benchmark...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(bench_df["batch_size"], bench_df["sklearn_predict_proba_seconds"] * 1000,
                 marker="o", color=BASELINE_COLOR, label="sklearn predict_proba")
    axes[0].plot(bench_df["batch_size"], bench_df["numpy_forward_pass_seconds"] * 1000,
                 marker="o", color=NN_FILTER_COLOR, label="numpy forward pass")
    axes[0].set_xlabel("Bins scored per call (batch size)")
    axes[0].set_ylabel("Time per call (ms)")
    axes[0].set_title("Per-call inference time")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    axes[1].bar(bench_df["batch_size"].astype(str), bench_df["speedup_factor"], color=NN_FILTER_COLOR)
    axes[1].set_xlabel("Bins scored per call (batch size)")
    axes[1].set_ylabel("Speedup (x times faster)")
    axes[1].set_title("Numpy speedup over sklearn")
    for i, v in enumerate(bench_df["speedup_factor"]):
        axes[1].text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=9)

    fig.suptitle("HA2 NN: sklearn predict_proba vs raw numpy forward pass", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_inference_benchmark.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 9 (NEW): FILTERING FUNNEL
# ============================================================

def plot_filtering_funnel(stats_df):
    print("Plotting NN filtering funnel...")

    row = stats_df.iloc[0]
    categories = ["Bins screened", "Rejected by NN", "NN-approved but\ngeometry failed", "Items skipped\n(<=1 open bin)"]
    values = [
        row["bins_screened"],
        row["bins_rejected_by_nn"],
        row["nn_approved_but_geometry_failed"],
        row["skipped_single_bin_items"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(categories, values, color=[NN_FILTER_COLOR, "#C44E52", "#937860", OTHER_TIME_COLOR])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Count")
    ax.set_title("HA2 NN: Filtering Outcomes")
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "ha2_filtering_funnel.png")
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    os.makedirs(VISUALS_FOLDER, exist_ok=True)

    ensure_results_exist()
    results = load_results()

    plot_hyperparameter_tuning(results["grid"])
    plot_variable_importance(results["importance"])
    plot_roc_and_threshold(results["roc"], results["sweep"], results["confusion"])
    plot_confusion_matrix(results["confusion"])
    plot_bucket_comparison(results["baseline"], results["nn_filtered"])
    plot_time_vs_length(results["baseline"], results["nn_filtered"])
    plot_threshold_sweep(results["nn_sweep"])
    plot_time_breakdown(results["nn_filtered"], results["baseline"])
    plot_inference_benchmark(results["benchmark"])
    plot_filtering_funnel(results["filtering_stats"])

    print(f"\nAll plots saved in the '{VISUALS_FOLDER}' folder.")


if __name__ == "__main__":
    main()