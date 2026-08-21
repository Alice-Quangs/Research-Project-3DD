# analyze1.py
#
# Reads all the .csv files ha1_nn.py saved into the "Results" folder and
# turns them into plots, saved into a "Visuals" folder (created
# automatically). This script does NOT do any packing or training itself
# - it just reads numbers and draws pictures. If the Results files don't
# exist yet, it runs ha1_nn.py first to generate them.
#
# Plots produced:
#   1. Hyperparameter tuning plot   (accuracy + kappa, per size/decay)
#   2. Variable importance          (which features the NN relies on)
#   3. ROC curve + threshold choice (how we picked the decision threshold)
#   4. Confusion matrix             (TP / FP / TN / FN at that threshold)
#   5. Bucketed comparison          (baseline vs NN-filtered, by order size)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FOLDER = "Results"
VISUALS_FOLDER = "Visuals"

# Same order-size buckets used throughout the project, based on the
# BED-BPP dataset's distribution of order sizes.
BUCKET_EDGES = [0, 20, 30, 40, 50, 60, 80, float("inf")]
BUCKET_LABELS = ["1-20", "21-30", "31-40", "41-50", "51-60", "61-80", "81+"]

BASELINE_COLOR = "#4C72B0"    # blue
NN_FILTER_COLOR = "#DD8452"   # orange


# ============================================================
#  MAKING SURE THE RESULTS FILES ACTUALLY EXIST
# ============================================================

REQUIRED_RESULTS_FILES = [
    "ha1_grid_search_results.csv",
    "ha1_variable_importance.csv",
    "ha1_roc_curve.csv",
    "ha1_threshold_sweep.csv",
    "ha1_confusion_matrix.csv",
    "ha1_baseline_per_order.csv",
    "ha1_nn_filtered_per_order.csv",
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
        print("Running ha1_nn.py first to generate them (this trains the NN and")
        print("runs the heuristic across the whole dataset, so it will take a while)...\n")

        import ha1_nn
        ha1_nn.main()
    else:
        print("All required Results files found - skipping straight to plotting.")


def load_results():
    def read(name):
        return pd.read_csv(os.path.join(RESULTS_FOLDER, name))

    return {
        "grid": read("ha1_grid_search_results.csv"),
        "importance": read("ha1_variable_importance.csv"),
        "roc": read("ha1_roc_curve.csv"),
        "sweep": read("ha1_threshold_sweep.csv"),
        "confusion": read("ha1_confusion_matrix.csv"),
        "baseline": read("ha1_baseline_per_order.csv"),
        "nn_filtered": read("ha1_nn_filtered_per_order.csv"),
    }


# ============================================================
#  PLOT 1: HYPERPARAMETER TUNING (ACCURACY + KAPPA)
# ============================================================
#
# One line per decay value, showing how accuracy/kappa changed as the
# hidden layer size grew.

def plot_hyperparameter_tuning(grid_df):
    print("Plotting hyperparameter tuning results...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for decay_value in sorted(grid_df["decay"].unique()):
        subset = grid_df[grid_df["decay"] == decay_value].sort_values("hidden_layer_size")
        axes[0].plot(subset["hidden_layer_size"], subset["cv_accuracy"],
                     marker="o", label=f"decay={decay_value}")
        axes[1].plot(subset["hidden_layer_size"], subset["cv_kappa"],
                     marker="o", label=f"decay={decay_value}")

    axes[0].set_title("Cross-Validated Accuracy")
    axes[0].set_xlabel("Hidden layer size")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(title="Decay", fontsize=8)

    axes[1].set_title("Cross-Validated Kappa")
    axes[1].set_xlabel("Hidden layer size")
    axes[1].set_ylabel("Kappa")
    axes[1].legend(title="Decay", fontsize=8)

    fig.suptitle("HA1 NN Hyperparameter Tuning", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "hyperparameter_tuning.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    ax.set_title("HA1 NN Variable Importance")
    fig.tight_layout()

    save_path = os.path.join(VISUALS_FOLDER, "variable_importance.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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

    # --- Panel 1: ROC curve ---
    axes[0].plot(roc_df["fpr"], roc_df["tpr"], color=BASELINE_COLOR, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

    # Mark where the chosen threshold sits on the sweep data
    chosen_row = sweep_df.iloc[(sweep_df["threshold"] - chosen_threshold).abs().idxmin()]
    axes[0].scatter([chosen_row["fpr"]], [chosen_row["tpr"]], color=NN_FILTER_COLOR, zorder=5,
                     label=f"Chosen threshold = {chosen_threshold:.2f}")

    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)

    # Panel 2: how the threshold was chosen (Youden's J across thresholds) ---
    axes[1].plot(sweep_df["threshold"], sweep_df["youden_j"], color=BASELINE_COLOR, label="Youden's J (TPR - FPR)")
    axes[1].axvline(chosen_threshold, color=NN_FILTER_COLOR, linestyle="--",
                     label=f"Chosen threshold = {chosen_threshold:.2f}")
    axes[1].set_title("Threshold Selection")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Youden's J")
    axes[1].legend(fontsize=8)

    fig.suptitle("HA1 NN: ROC Curve and Threshold Choice", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "roc_and_threshold.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
#  PLOT 4: CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(confusion_df):
    print("Plotting confusion matrix...")

    row = confusion_df.iloc[0]
    # matrix laid out as: rows = actual class, columns = predicted class
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

    # Write the actual count in the middle of each square
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    footnote = (f"Precision={row['precision']:.3f}  Recall={row['recall']:.3f}  "
                f"F1={row['f1_score']:.3f}  Accuracy={row['accuracy']:.3f}  Kappa={row['kappa']:.3f}")
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=9)

    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "confusion_matrix.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    ).reset_index()
    summary["source"] = label
    return summary


def plot_bucket_comparison(baseline_df, nn_df):
    print("Plotting bucketed baseline vs NN-filtered comparison...")

    baseline_summary = bucket_and_summarize(baseline_df, "Baseline")
    nn_summary = bucket_and_summarize(nn_df, "NN filter")
    combined = pd.concat([baseline_summary, nn_summary], ignore_index=True)

    print("\n" + "=" * 80)
    print(" HA1 PERFORMANCE BY ORDER SCALE (BASELINE vs NN FILTER)")
    print("=" * 80)
    print(combined.to_string(index=False))

    pivot_bins = combined.pivot(index="scale_bucket", columns="source", values="avg_bins_per_order")
    pivot_util = combined.pivot(index="scale_bucket", columns="source", values="avg_utilization_pct")
    pivot_cost = combined.pivot(index="scale_bucket", columns="source", values="avg_cost")
    pivot_time = combined.pivot(index="scale_bucket", columns="source", values="avg_time_seconds")

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

    pivot_time.plot(kind="bar", ax=axes[1, 1], color=colors)
    axes[1, 1].set_title("Avg Time per Order")
    axes[1, 1].set_xlabel("Order size (items)")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].tick_params(axis="x", rotation=0)

    fig.suptitle("HA1: Baseline vs NN Filter, by Order Scale", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(VISUALS_FOLDER, "scale_comparison.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {save_path}")

    # Print a quick time-savings summary to the console, same idea as before.
    print("\n" + "=" * 80)
    print(" TIME SAVINGS SUMMARY (NN filter vs Baseline)")
    print("=" * 80)
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

    print(f"\nAll plots saved in the '{VISUALS_FOLDER}' folder.")


if __name__ == "__main__":
    main()