# analyze_ha2_by_scale.py
#
# Buckets orders by item count using Option A buckets (based on BED-BPP distribution):
#   1-20, 21-30, 31-40, 41-50, 51-60, 61-80, 81+
#
# Compares HA2 baseline vs HA2+NN filter performance WITHIN each bucket.
# Shows: avg bins per order, avg utilization, and avg time per order.
#
# Reads: ha2_baseline_per_order.csv, ha2_nn_filtered_per_order.csv
# (both produced by their respective scripts, now with time_seconds column)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("Visuals", exist_ok=True)

BASELINE_PATH = "ha2_baseline_per_order.csv"
NN_FILTERED_PATH = "ha2_nn_filtered_per_order.csv"

# Option A buckets based on BED-BPP distribution
BUCKET_EDGES = [0, 20, 30, 40, 50, 60, 80, float("inf")]
BUCKET_LABELS = ["1-20", "21-30", "31-40", "41-50", "51-60", "61-80", "81+"]


def bucket_and_summarize(df, label):
    df = df.copy()
    df["scale_bucket"] = pd.cut(
        df["item_count"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True
    )

    summary = df.groupby("scale_bucket", observed=True).agg(
        n_orders=("order_id", "count"),
        avg_bins_per_order=("bins_used", "mean"),
        avg_utilization_pct=("utilization_pct", "mean"),
        avg_cost=("cost", "mean"),
        avg_time_seconds=("time_seconds", "mean"),
    ).reset_index()
    summary["source"] = label
    return summary


def main():
    baseline_df = pd.read_csv(BASELINE_PATH)
    nn_df = pd.read_csv(NN_FILTERED_PATH)

    baseline_summary = bucket_and_summarize(baseline_df, "Baseline")
    nn_summary = bucket_and_summarize(nn_df, "NN filter")

    combined = pd.concat([baseline_summary, nn_summary], ignore_index=True)

    print("\n" + "=" * 80)
    print(" HA2 PERFORMANCE BY ORDER SCALE (BASELINE vs NN FILTER)")
    print("=" * 80)
    print(combined.to_string(index=False))

    combined.to_csv("ha2_scale_comparison.csv", index=False)
    print("\nSaved: ha2_scale_comparison.csv")

    # Wide format for plotting: one row per bucket, columns for each source
    pivot_bins = combined.pivot(index="scale_bucket", columns="source", values="avg_bins_per_order")
    pivot_util = combined.pivot(index="scale_bucket", columns="source", values="avg_utilization_pct")
    pivot_time = combined.pivot(index="scale_bucket", columns="source", values="avg_time_seconds")

    # Create 3-panel chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Avg Bins per Order
    pivot_bins.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Avg Bins per Order by Scale")
    axes[0].set_xlabel("Order size (items)")
    axes[0].set_ylabel("Avg bins used")
    axes[0].legend(title="")
    axes[0].tick_params(axis='x', rotation=0)

    # Panel 2: Avg Space Utilization
    pivot_util.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Avg Space Utilization by Scale")
    axes[1].set_xlabel("Order size (items)")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].legend(title="")
    axes[1].tick_params(axis='x', rotation=0)

    # Panel 3: Avg Time per Order
    pivot_time.plot(kind="bar", ax=axes[2], color=["#4C72B0", "#DD8452"])
    axes[2].set_title("Avg Time per Order by Scale")
    axes[2].set_xlabel("Order size (items)")
    axes[2].set_ylabel("Time (seconds)")
    axes[2].legend(title="")
    axes[2].tick_params(axis='x', rotation=0)

    fig.suptitle("HA2: Baseline vs NN Filter, by Order Scale", fontsize=14)
    fig.tight_layout()
    save_path = "Visuals/ha2_scale_comparison_chart.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)

    # Print time savings summary
    print("\n" + "=" * 80)
    print(" TIME SAVINGS SUMMARY")
    print("=" * 80)
    for bucket in BUCKET_LABELS:
        baseline_time = baseline_summary[baseline_summary["scale_bucket"] == bucket]["avg_time_seconds"].values[0]
        nn_time = nn_summary[nn_summary["scale_bucket"] == bucket]["avg_time_seconds"].values[0]
        n_orders = baseline_summary[baseline_summary["scale_bucket"] == bucket]["n_orders"].values[0]
        savings = ((baseline_time - nn_time) / baseline_time * 100) if baseline_time > 0 else 0
        print(f"  {bucket:>8} items ({n_orders:>5} orders): Baseline={baseline_time:.4f}s, NN={nn_time:.4f}s, Savings={savings:.1f}%")


if __name__ == "__main__":
    main()