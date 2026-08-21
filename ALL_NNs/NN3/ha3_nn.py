# ha3_nn.py
#
# This one script does the whole HA3 (Extreme Points heuristic) machine
# learning job, start to finish:
#
#   1. Build training data straight from the raw BED-BPP dataset.
#   2. Train a small neural network (an MLP - Multi Layer Perceptron) to
#      predict Fit / No-Fit, using a grid search + 5-fold cross-validation,
#      the same way the dissertation did it.
#   3. Work out the best decision threshold for the NN using a threshold
#      sweep (this is what the ROC curve is built from).
#   4. Run the plain HA3 heuristic on the whole dataset (the "baseline").
#   5. Run HA3 again, but this time the NN gets to reject bad-looking bins
#      before the heuristic bothers checking their geometry (the
#      "NN-filtered" run).
#
# Every result gets saved as a .csv file into a "Results" folder (created
# automatically). analyze3.py then reads those .csv files and turns them
# into plots. This script does NOT make any plots itself.
#
# This script does not change ha3.py or bin_types.py at all - it only
# imports functions from them.

import os
import json
import csv
import time
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    roc_curve, roc_auc_score, make_scorer,
)
from sklearn.inspection import permutation_importance

from bin_types import bin_types
from ha3 import (
    Bin,
    all_orientations,
    can_place,
    update_ep,
    best_orientation,
    run_ep_method,
)


# ============================================================
#  SETTINGS 
# ============================================================

DATA_PATH = "../dataset/bed-bpp_v1.json" # the data

RESULTS_FOLDER = "Results"

RANDOM_STATE = 42          # fixed seed so results are repeatable

# Grid searching 20 hyperparameter combinations, each with 5-fold CV, on
# the FULL dataset (potentially tens of millions of rows) would take an
# extremely long time. So the grid search itself runs on a smaller,
# stratified sample. Once we know the best hyperparameters, we train the
# FINAL model on the full dataset.
GRID_SEARCH_SAMPLE_SIZE = 300_000

#  size/decay grid 
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(1,), (3,), (5,), (7,), (9,)],
    "mlp__alpha": [0.0, 0.0001, 0.001, 0.01],
}

# The 19 numbers the NN looks at for every (bin, item) decision.
# First 7 are the original "how big is the bin/item, how much space is
# left" features. The next 12 describe the SHAPE of the bin's remaining
# Extreme Points (EP) - the same EP concept the heuristic itself uses to
# decide where to place items.
#
# Unlike an EMS, a bare Extreme Point is just a single (x, y, z) location -
# it doesn't carry its own "how much room is here" size the way an EMS
# does. So instead of max/min EMS volume, we describe the two EPs that
# matter most to a first-fit-style heuristic: the one CLOSEST to the bin's
# origin corner (the one the heuristic will try first) and the one
# FURTHEST from it (a rough gauge of how far the packing has spread out).
# For the closest EP we also add its "headroom" - the raw distance from
# that point to the three far bin walls - as a stand-in for how much room
# is actually left to place something there.
FEATURES = [
    "bin_length", "bin_width", "bin_height",
    "item_length", "item_width", "item_height",
    "red_cap",
    "closest_ep_distance", "closest_ep_x", "closest_ep_y", "closest_ep_z",
    "closest_ep_headroom_x", "closest_ep_headroom_y", "closest_ep_headroom_z",
    "furthest_ep_distance", "furthest_ep_x", "furthest_ep_y", "furthest_ep_z",
    "ep_count",
]
TARGET = "fit"


# ============================================================
#  SMALL HELPER FUNCTIONS
# ============================================================

def load_dataset(path):
    print(f"Loading dataset from {path} ...")
    with open(path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} orders")
    return dataset


def build_sequence(order):
    """
    Turns one order's raw JSON item_sequence into a simple, ordered list
    of dicts: [{'length':.., 'width':.., 'height':..}, ...]
    Items are sorted by their original sequence number so they arrive in
    the same order they did in the real order.
    """
    items = []
    for key in sorted(order["item_sequence"].keys(), key=lambda x: int(x)):
        item = order["item_sequence"][key]
        items.append({
            "length": item["length/mm"],
            "width": item["width/mm"],
            "height": item["height/mm"],
        })
    return items


def bin_red_cap(bin_obj):
    """Residual capacity: how much empty volume is left in this bin right now."""
    bin_volume = bin_obj.length * bin_obj.width * bin_obj.height
    used_volume = sum(
        (x2 - x1) * (y2 - y1) * (z2 - z1)
        for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
    )
    return bin_volume - used_volume


def get_ep_stats(ep_list, bin_l, bin_w, bin_h):
    """
    Summarises a bin's list of Extreme Points (EP).

    red_cap alone only tells you the TOTAL empty volume - it can't tell
    the difference between a bin with one big EP sitting right at the
    origin corner and another bin with the exact same red_cap but every
    EP scattered far away in awkward spots. These EP stats give the NN
    that missing information: where the most promising point is, how far
    the packing has spread out, and roughly how much room is left around
    the best point.

    Each EP is stored as (x, y, z). Distance is measured from the bin's
    origin corner (0, 0, 0) - the same corner the heuristic always starts
    searching from. Returns a tuple of 12 numbers:
        closest_distance, closest_x, closest_y, closest_z,
        closest_headroom_x, closest_headroom_y, closest_headroom_z,
        furthest_distance, furthest_x, furthest_y, furthest_z,
        count
    """
    if not ep_list:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    distances = [math.sqrt(x ** 2 + y ** 2 + z ** 2) for (x, y, z) in ep_list]

    closest_i = distances.index(min(distances))
    furthest_i = distances.index(max(distances))

    cx, cy, cz = ep_list[closest_i]
    fx, fy, fz = ep_list[furthest_i]

    # Headroom: how far the closest EP is from the three "far" bin walls -
    # a rough stand-in for how much room is actually left to use there.
    headroom_x = bin_l - cx
    headroom_y = bin_w - cy
    headroom_z = bin_h - cz

    return (
        distances[closest_i], cx, cy, cz,
        headroom_x, headroom_y, headroom_z,
        distances[furthest_i], fx, fy, fz,
        len(ep_list),
    )


def make_feature_row(bin_obj, item_l, item_w, item_h):
    """Builds one 19-number feature row for a single (bin, item) pairing."""
    red_cap = bin_red_cap(bin_obj)
    ep_stats = get_ep_stats(bin_obj.ep_list, bin_obj.length, bin_obj.width, bin_obj.height)
    return [
        bin_obj.length, bin_obj.width, bin_obj.height,
        item_l, item_w, item_h,
        red_cap,
        *ep_stats,
    ]


# ============================================================
#  STAGE 1 - GENERATE TRAINING DATA
# ============================================================
#
# This re-does exactly what HA3's first-fit EP packing logic does, one
# item at a time, but ALSO writes down a training row every time it asks
# "does this item fit in this open bin?" - which is precisely the
# question the NN will later be asked to answer.
#
# Note: every feature is measured on the bin's state BEFORE the item is
# placed (i.e. "given what this bin looks like right now, will this item
# fit?"), which matches exactly what the NN will be asked at prediction
# time later on.

def pack_item_with_logging(open_bins, item_l, item_w, item_h, log_rows):
    orientations = all_orientations(item_l, item_w, item_h)

    for bin_obj in open_bins:
        # Record the feature row BEFORE we touch this bin.
        feature_row = make_feature_row(bin_obj, item_l, item_w, item_h)

        placed = False
        for ep_x, ep_y, ep_z in bin_obj.ep_list:
            valid_orientations = [
                (ol, ow, oh) for (ol, ow, oh) in orientations
                if can_place(ep_x, ep_y, ep_z, ol, ow, oh, bin_obj.length, bin_obj.width, bin_obj.height, bin_obj.placed_items)
            ]
            if valid_orientations:
                # This is the first point (in EP sort order) that works -
                # place the item here and stop, same as the real heuristic.
                ol, ow, oh = best_orientation(valid_orientations)

                bin_obj.placed_items.append((ep_x, ep_y, ep_z, ep_x + ol, ep_y + ow, ep_z + oh))
                bin_obj.ep_list = update_ep(bin_obj.ep_list, ep_x, ep_y, ep_z, ol, ow, oh,
                                            bin_obj.placed_items, bin_obj.length, bin_obj.width, bin_obj.height)
                placed = True
                break

        fits_here = 1 if placed else 0
        log_rows.append(feature_row + [fits_here])

        if placed:
            return bin_obj

    # None of the open bins worked - open a brand new one.
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)

                # Log the feature row for this new bin BEFORE placing the
                # item, i.e. while it's still one single EP at the origin
                # corner of the whole bin. It always "fits" here by
                # definition, since we only reach this branch when the
                # item is small enough for this bin_type.
                log_rows.append(make_feature_row(new_bin, item_l, item_w, item_h) + [1])

                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, ol, ow, oh,
                                            new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
                open_bins.append(new_bin)
                return new_bin

    print("Warning: an item could not be packed into any bin type!")
    return None


def generate_training_data(dataset):
    print("\n" + "=" * 60)
    print(" STAGE 1: GENERATING TRAINING DATA")
    print("=" * 60)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    log_rows = []

    start_time = time.perf_counter()

    for idx, order_id in enumerate(order_ids):
        items = build_sequence(dataset[order_id])
        open_bins = []
        for item in items:
            pack_item_with_logging(open_bins, item["length"], item["width"], item["height"], log_rows)

        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders, {len(log_rows):,} rows so far ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"Done. Generated {len(log_rows):,} rows in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    df = pd.DataFrame(log_rows, columns=FEATURES + [TARGET])

    fit_count = int(df[TARGET].sum())
    print(f"Fit: {fit_count:,}   No Fit: {len(df) - fit_count:,}   "
          f"({fit_count / len(df) * 100:.1f}% Fit)")

    save_path = os.path.join(RESULTS_FOLDER, "ha3_training_data.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    return df


# ============================================================
#  STAGE 2 - TRAIN THE NEURAL NETWORK
# ============================================================

def train_neural_network(df):
    print("\n" + "=" * 60)
    print(" STAGE 2: TRAINING THE NEURAL NETWORK")
    print("=" * 60)

    X = df[FEATURES]
    y = df[TARGET]

    # Hold out 20% of the FULL dataset as a test set. This is set aside
    # right at the start and is never used for training anything - not
    # the grid search, not the final model - so every evaluation number
    # we report later (accuracy, kappa, ROC, confusion matrix, variable
    # importance) is measured fairly, on data the model has never seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training rows: {len(X_train):,}   Held-out test rows: {len(X_test):,}")

    # --- Grid search on a smaller, stratified sample of the training set ---
    # (Only for SPEED - trying 20 hyperparameter combinations x 5-fold CV
    # on tens of millions of rows would take an extremely long time. The
    # final model below is still trained on the FULL training set.)
    if len(X_train) > GRID_SEARCH_SAMPLE_SIZE:
        X_grid, _, y_grid, _ = train_test_split(
            X_train, y_train,
            train_size=GRID_SEARCH_SAMPLE_SIZE,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_grid, y_grid = X_train, y_train
    print(f"Grid search sample: {len(X_grid):,} rows")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="logistic",   # sigmoid activation, matches the dissertation
            solver="lbfgs",          # a BFGS-family optimiser
                                      # methodology (R's nnet also optimises this way by
                                      # default). Backpropagation is still what computes the
                                      # gradients that this optimiser uses at every step -
                                      # sklearn does that internally regardless of solver.
            max_iter=300,            #  iteration cap
            random_state=RANDOM_STATE,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "kappa": make_scorer(cohen_kappa_score),
    }

    print("\nRunning grid search (5-fold cross-validation, 20 combinations)...")
    print("This trains the network many times over via backpropagation - it will take a while.")
    grid_start = time.perf_counter()
    grid = GridSearchCV(
        pipeline, PARAM_GRID,
        scoring=scoring, refit="accuracy",
        cv=cv, n_jobs=-1, verbose=1,
    )
    grid.fit(X_grid, y_grid)
    grid_time = time.perf_counter() - grid_start
    print(f"Grid search finished in {grid_time:.1f}s ({grid_time / 60:.2f} min)")

    results = pd.DataFrame(grid.cv_results_)
    results["hidden_layer_size"] = results["param_mlp__hidden_layer_sizes"].apply(lambda t: t[0])
    results["decay"] = results["param_mlp__alpha"]
    grid_results_table = results[
        ["hidden_layer_size", "decay", "mean_test_accuracy", "mean_test_kappa"]
    ].rename(columns={"mean_test_accuracy": "cv_accuracy", "mean_test_kappa": "cv_kappa"}
    ).sort_values(["hidden_layer_size", "decay"])

    grid_path = os.path.join(RESULTS_FOLDER, "ha3_grid_search_results.csv")
    grid_results_table.to_csv(grid_path, index=False)
    print(f"Saved: {grid_path}")
    print(f"Best hyperparameters found: {grid.best_params_}")

    #  Fit the FINAL model on the FULL training set 
    best_size = grid.best_params_["mlp__hidden_layer_sizes"]
    best_decay = grid.best_params_["mlp__alpha"]

    print(f"\nTraining final model on the full training set ({len(X_train):,} rows) "
          f"using size={best_size[0]}, decay={best_decay} ...")
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=best_size,
            alpha=best_decay,
            activation="logistic",
            solver="lbfgs",
            max_iter=300,
            random_state=RANDOM_STATE,
        )),
    ])
    fit_start = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    print(f"Final model trained in {fit_time:.1f}s ({fit_time / 60:.2f} min)")

    #  Evaluate on the held-out test set 
    y_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_kappa = cohen_kappa_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {test_accuracy:.4f}   Kappa: {test_kappa:.4f}")

    # --- Variable importance ---
    # sklearn's MLP has no built-in importance score (unlike the Garson
    # method used with R's nnet), so we use permutation importance
    # instead: it shuffles one feature at a time and measures how much
    # worse the model gets, which tells us how much that feature matters.
    print("\nComputing permutation importance (this re-runs predictions many times)...")
    perm_result = permutation_importance(
        final_model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    # scaled 0-100, same style as the dissertation's importance plots
    importance_df["importance_scaled"] = (
        importance_df["importance_mean"] / importance_df["importance_mean"].max() * 100
    )
    importance_path = os.path.join(RESULTS_FOLDER, "ha3_variable_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")

    #  ROC curve data 
    fit_class_index = list(final_model.classes_).index(1)
    test_probs = final_model.predict_proba(X_test)[:, fit_class_index]

    fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
    roc_auc = roc_auc_score(y_test, test_probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    roc_df["roc_auc"] = roc_auc  # same value repeated on every row, for convenience
    roc_path = os.path.join(RESULTS_FOLDER, "ha3_roc_curve.csv")
    roc_df.to_csv(roc_path, index=False)
    print(f"Saved: {roc_path}  (ROC AUC = {roc_auc:.4f})")

    # --- Threshold sweep ---
    # "Fit" is the rare class (a small % of rows), so the default 0.5
    # cutoff tends to miss most real Fits. We test a wide range of
    # thresholds and, for each one, work out precision/recall/etc, then
    # pick the threshold using Youden's J statistic: J = TPR - FPR. This
    # is the point on the ROC curve furthest above the "random guess"
    # diagonal line - a standard, explainable way to pick a threshold.
    print("\nSweeping decision thresholds to choose the best one...")
    thresholds_to_test = np.arange(0.01, 1.00, 0.01)
    sweep_rows = []
    for t in thresholds_to_test:
        preds_at_t = (test_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_at_t, labels=[0, 1]).ravel()

        tpr_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_t = tpr_t
        f1_t = (2 * precision_t * recall_t / (precision_t + recall_t)
                if (precision_t + recall_t) > 0 else 0.0)
        youden_j = tpr_t - fpr_t

        sweep_rows.append([t, tp, fp, tn, fn, precision_t, recall_t, f1_t, tpr_t, fpr_t, youden_j])

    sweep_df = pd.DataFrame(sweep_rows, columns=[
        "threshold", "tp", "fp", "tn", "fn",
        "precision", "recall", "f1", "tpr", "fpr", "youden_j",
    ])
    sweep_path = os.path.join(RESULTS_FOLDER, "ha3_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved: {sweep_path}")

    best_row = sweep_df.loc[sweep_df["youden_j"].idxmax()]
    chosen_threshold = float(best_row["threshold"])
    print(f"Chosen threshold (max Youden's J): {chosen_threshold:.2f}  "
          f"(TPR={best_row['tpr']:.3f}, FPR={best_row['fpr']:.3f})")

    #  Confusion matrix + summary metrics at the chosen threshold 
    final_preds = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy_at_threshold = (tp + tn) / (tp + tn + fp + fn)
    kappa_at_threshold = cohen_kappa_score(y_test, final_preds)

    confusion_df = pd.DataFrame([{
        "chosen_threshold": chosen_threshold,
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "precision": precision, "recall": recall, "f1_score": f1,
        "accuracy": accuracy_at_threshold, "kappa": kappa_at_threshold,
        "roc_auc": roc_auc,
    }])
    confusion_path = os.path.join(RESULTS_FOLDER, "ha3_confusion_matrix.csv")
    confusion_df.to_csv(confusion_path, index=False)
    print(f"Saved: {confusion_path}")

    # Save the trained model itself (not a .csv - it's needed by Stage 4
    # below, and if you re-run this script later you can load it back
    # instead of re-training from scratch).
    model_path = "ha3_nn_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved: {model_path}")

    return final_model, chosen_threshold


# ============================================================
#  STAGE 3 - RUN THE PLAIN HEURISTIC (BASELINE)
# ============================================================
#
# This uses run_ep_method directly - the real heuristic function from
# ha3.py - so these numbers are the ground truth to compare everything
# else against.

def run_baseline(dataset):
    print("\n" + "=" * 60)
    print(" STAGE 3: RUNNING THE PLAIN HEURISTIC (BASELINE)")
    print("=" * 60)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    per_order_rows = []

    start_time = time.perf_counter()

    for idx, order_id in enumerate(order_ids):
        sequence = build_sequence(dataset[order_id])

        order_start = time.perf_counter()
        open_bins, num_bins, cost = run_ep_method(sequence, bin_types, visualize=False)
        order_time = time.perf_counter() - order_start

        volume_used = sum(
            (x2 - x1) * (y2 - y1) * (z2 - z1)
            for bin_obj in open_bins
            for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
        )
        bin_volume_total = sum(b.length * b.width * b.height for b in open_bins)
        utilization = (volume_used / bin_volume_total * 100) if bin_volume_total else 0

        per_order_rows.append([order_id, len(sequence), num_bins, cost, utilization, order_time])

        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"Baseline run finished in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    per_order_df = pd.DataFrame(per_order_rows, columns=[
        "order_id", "item_count", "bins_used", "cost", "utilization_pct", "time_seconds",
    ])
    save_path = os.path.join(RESULTS_FOLDER, "ha3_baseline_per_order.csv")
    per_order_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    return per_order_df


# ============================================================
#  STAGE 4 - RUN THE NN-FILTERED HEURISTIC
# ============================================================
#
# Same first-fit EP packing as before, but this time, before the
# heuristic bothers checking a bin's exact geometry, the NN gets a quick
# look at all the currently open bins and rejects the ones it thinks are
# bad candidates. Only the bins the NN approves of go through the full
# (slower) geometric check.

def nn_predict_all_bins(model, threshold, open_bins, item_l, item_w, item_h):
    """
    Asks the NN about every open bin for this item in ONE batched call
    (much faster than calling the model once per bin). Returns a list of
    True/False - True meaning "NN says this bin is worth checking properly".
    """
    if not open_bins:
        return []

    rows = [make_feature_row(bin_obj, item_l, item_w, item_h) for bin_obj in open_bins]
    batch = pd.DataFrame(rows, columns=FEATURES)

    fit_class_index = list(model.classes_).index(1)
    probs = model.predict_proba(batch)[:, fit_class_index]
    return [p >= threshold for p in probs]


def pack_item_nn_filtered(open_bins, item_l, item_w, item_h, model, threshold, stats):
    orientations = all_orientations(item_l, item_w, item_h)

    nn_says_worth_checking = nn_predict_all_bins(model, threshold, open_bins, item_l, item_w, item_h)
    stats["bins_screened"] += len(open_bins)

    for bin_obj, nn_approved in zip(open_bins, nn_says_worth_checking):
        if not nn_approved:
            stats["bins_rejected_by_nn"] += 1
            continue  # skip the slow geometric check entirely for this bin

        placed = False
        for ep_x, ep_y, ep_z in bin_obj.ep_list:
            valid_orientations = [
                (ol, ow, oh) for (ol, ow, oh) in orientations
                if can_place(ep_x, ep_y, ep_z, ol, ow, oh, bin_obj.length, bin_obj.width, bin_obj.height, bin_obj.placed_items)
            ]
            if valid_orientations:
                ol, ow, oh = best_orientation(valid_orientations)
                bin_obj.placed_items.append((ep_x, ep_y, ep_z, ep_x + ol, ep_y + ow, ep_z + oh))
                bin_obj.ep_list = update_ep(bin_obj.ep_list, ep_x, ep_y, ep_z, ol, ow, oh,
                                            bin_obj.placed_items, bin_obj.length, bin_obj.width, bin_obj.height)
                placed = True
                break

        if placed:
            return bin_obj
        else:
            # The NN approved this bin but the real geometry disagreed.
            # This can happen because the NN only sees summary numbers
            # about the bin, not the exact position of every EP.
            stats["nn_approved_but_geometry_failed"] += 1
            continue

    # Nothing worked - open a new bin, same as the baseline.
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, ol, ow, oh,
                                            new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
                open_bins.append(new_bin)
                return new_bin

    print("Warning: an item could not be packed into any bin type!")
    return None


def run_nn_filtered(dataset, model, threshold):
    print("\n" + "=" * 60)
    print(" STAGE 4: RUNNING THE NN-FILTERED HEURISTIC")
    print("=" * 60)
    print(f"Using decision threshold: {threshold:.2f}")

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    per_order_rows = []
    stats = {"bins_screened": 0, "bins_rejected_by_nn": 0, "nn_approved_but_geometry_failed": 0}

    start_time = time.perf_counter()

    for idx, order_id in enumerate(order_ids):
        items = build_sequence(dataset[order_id])
        open_bins = []

        order_start = time.perf_counter()
        for item in items:
            pack_item_nn_filtered(
                open_bins, item["length"], item["width"], item["height"],
                model, threshold, stats,
            )
        order_time = time.perf_counter() - order_start

        num_bins = len(open_bins)
        cost = sum(b.cost for b in open_bins)
        volume_used = sum(
            (x2 - x1) * (y2 - y1) * (z2 - z1)
            for bin_obj in open_bins
            for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
        )
        bin_volume_total = sum(b.length * b.width * b.height for b in open_bins)
        utilization = (volume_used / bin_volume_total * 100) if bin_volume_total else 0

        per_order_rows.append([order_id, len(items), num_bins, cost, utilization, order_time])

        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"NN-filtered run finished in {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"Bins screened: {stats['bins_screened']:,}   "
          f"Rejected by NN: {stats['bins_rejected_by_nn']:,} "
          f"({stats['bins_rejected_by_nn'] / max(stats['bins_screened'], 1) * 100:.1f}%)   "
          f"NN-approved but geometry failed anyway: {stats['nn_approved_but_geometry_failed']:,}")

    per_order_df = pd.DataFrame(per_order_rows, columns=[
        "order_id", "item_count", "bins_used", "cost", "utilization_pct", "time_seconds",
    ])
    save_path = os.path.join(RESULTS_FOLDER, "ha3_nn_filtered_per_order.csv")
    per_order_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(RESULTS_FOLDER, "ha3_nn_filtering_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    return per_order_df


# ============================================================
#  MAIN
# ============================================================

def main():
    script_start = time.perf_counter()
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    dataset = load_dataset(DATA_PATH)

    training_df = generate_training_data(dataset)
    model, chosen_threshold = train_neural_network(training_df)
    run_baseline(dataset)
    run_nn_filtered(dataset, model, chosen_threshold)

    total_time = time.perf_counter() - script_start
    print("\n" + "=" * 60)
    print(" ALL STAGES COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"All .csv results saved in the '{RESULTS_FOLDER}' folder.")


if __name__ == "__main__":
    main()