# ha3_nn_v2.py
#
# Full HA3 (Extreme Points) pipeline, end to end:
#
#   1. Build training data straight from the raw BED-BPP dataset.
#   2. Train the MLP (grid search + 5-fold CV), pick a decision threshold.
#   3. Run the plain HA3 heuristic on the whole dataset (the baseline).
#   4. Extract the trained model's weights and do the NN's job with a raw
#      numpy forward pass instead of sklearn's predict_proba - this is
#      the fast path. Sanity-checked against sklearn before it's trusted.
#   5. Run HA3 again with the NN gatekeeper (fast path), skipping the NN
#      entirely whenever 0-1 bins are open (nothing to screen/choose
#      between there), and logging how much time goes to NN inference vs
#      geometry checks vs everything else, per order.
#   6. Threshold sweep (also using the fast path) for the line graph.
#
# Everything gets saved as .csv (plus one .joblib model and one .npz
# weights file) into a "Results" folder. analyze3_v2.py reads those and
# makes the plots - this script does not draw anything itself.

import os
import json
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
from ha3 import Bin, all_orientations, can_place, update_ep, best_orientation, run_ep_method


# ============================================================
#  SETTINGS
# ============================================================

DATA_PATH = "../dataset/bed-bpp_v1.json"
RESULTS_FOLDER = "Results"
RANDOM_STATE = 42

GRID_SEARCH_SAMPLE_SIZE = 300_000

PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(1,), (3,), (5,), (7,), (9,)],
    "mlp__alpha": [0.0, 0.0001, 0.001, 0.01],
}

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

SWEEP_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
SWEEP_SUBSET_SIZE = 300


# ============================================================
#  SMALL HELPER FUNCTIONS
# ============================================================

def load_dataset(path):

    """
    Reads the whole dataset file from disk into memory as a python dictionary.
    """

    print(f"Loading dataset from {path} ...")
    with open(path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} orders")
    return dataset


def build_sequence(order):

    """
    Pulls one order's items out of the raw dataset and returns them as a plain
    list of {length, width, height} dictionaries in the exact order they arrived.
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


def make_feature_row(bin_obj, item_l, item_w, item_h):
    """Builds one 19-number feature row for a single (bin, item) pairing,
    reading from cached Bin attributes instead of recomputing."""
    return [
        bin_obj.length, bin_obj.width, bin_obj.height,
        item_l, item_w, item_h,
        bin_obj.red_cap,
        bin_obj.closest_ep_distance, bin_obj.closest_ep_x, bin_obj.closest_ep_y, bin_obj.closest_ep_z,
        bin_obj.closest_ep_headroom_x, bin_obj.closest_ep_headroom_y, bin_obj.closest_ep_headroom_z,
        bin_obj.furthest_ep_distance, bin_obj.furthest_ep_x, bin_obj.furthest_ep_y, bin_obj.furthest_ep_z,
        bin_obj.ep_count,
    ]


# bin_red_cap / get_ep_stats used to rescan placed_items / ep_list are updated
# incrementally by Bin itself (bin_types.py) - refreshed once per actual placement,
# not once per screening attempt - so feature-building above just reads the cached
# attributes (red_cap, closest_ep_distance, ...) directly. O(1) instead of O(EP count) per
#  bin per item screened.


# ============================================================
#  STAGE 1 - GENERATE TRAINING DATA
# ============================================================

def pack_item_with_logging(open_bins, item_l, item_w, item_h, log_rows):

    """
    For each bin already open, checks whether this item could be placed at
    one of its extreme points (trying every rotation, against every EP in
    turn). Logs ONE training row per bin checked (the bin/item features
    plus 1 (fits) or 0 (doesn't fit) ). This is how we get 'ground truth'
    labels without any model, just geometry.
    """

    orientations = all_orientations(item_l, item_w, item_h)

    for bin_obj in open_bins:
        feature_row = make_feature_row(bin_obj, item_l, item_w, item_h)

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
                bin_obj.register_placement(ol, ow, oh)
                placed = True
                break

        # 1 if the item fits at some extreme point in this bin, 0 if it
        # does not. This number is the label the model will later be
        # trained to predict.
        fits_here = 1 if placed else 0
        log_rows.append(feature_row + [fits_here])

        # If it fits here, place it now and stop checking other bins. This
        # is First Fit behaviour, the item goes into the first open bin
        # that can take it, not necessarily the best possible one.
        if placed:
            return bin_obj

    # Nothing already open could take this item, so open a new bin. Bin
    # types are ordered smallest first, so this picks the smallest size
    # that the item fits into in some rotation.
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                log_rows.append(make_feature_row(new_bin, item_l, item_w, item_h) + [1])
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, ol, ow, oh,
                                            new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
                new_bin.register_placement(ol, ow, oh)
                open_bins.append(new_bin)
                return new_bin

    # Should never actually happen, it would mean an item is bigger than
    # even the largest available bin type.
    print("Warning: an item could not be packed into any bin type!")
    return None


def generate_training_data(dataset):

    """
    Runs the whole dataset order-by-order. Each order starts with zero open bins (bins never carry
    over between orders). Every item packed generates one or more training rows via pack_item_with_logging.
    At the end, save everything as one big CSV — this is the training data Stage 2 will learn from
    """

    print("\n" + "=" * 60)
    print(" STAGE 1: GENERATING TRAINING DATA")
    print("=" * 60)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    log_rows = []

    start_time = time.perf_counter()
    for idx, order_id in enumerate(order_ids):
        # A fresh, empty set of open bins for every order. Bins from one
        # customer order are never reused for a different order.
        items = build_sequence(dataset[order_id])
        open_bins = []
        for item in items:
            pack_item_with_logging(open_bins, item["length"], item["width"], item["height"], log_rows)

        # Just a progress message every 500 orders so long runs are not silent.
        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders, {len(log_rows):,} rows so far ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"Done. Generated {len(log_rows):,} rows in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    # Turn the plain list of logged rows into a proper table, with column
    # names matching FEATURES plus the TARGET (fit) column.
    df = pd.DataFrame(log_rows, columns=FEATURES + [TARGET])
    fit_count = int(df[TARGET].sum())
    # A quick sanity check on how balanced the fit vs no fit labels are.
    print(f"Fit: {fit_count:,}   No Fit: {len(df) - fit_count:,}   ({fit_count / len(df) * 100:.1f}% Fit)")

    # Save the table to disk so Stage 2 can load it straight away, without
    # needing to re-run this whole simulation again.
    save_path = os.path.join(RESULTS_FOLDER, "ha3_training_data.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return df


# ============================================================
#  STAGE 2 - TRAIN THE NEURAL NETWORK
# ============================================================

def train_neural_network(df):

    """
    Takes the training table from Stage 1 and turns it into a trained model.
    Splits the data into train and test sets, searches for the best network
    size and regularisation strength using cross validation, trains the
    final model on the full training set, then picks the decision threshold
    (the cutoff probability above which we say "yes it fits") that gives
    the best balance between catching real fits and avoiding false alarms.
    """

    print("\n" + "=" * 60)
    print(" STAGE 2: TRAINING THE NEURAL NETWORK")
    print("=" * 60)

    X = df[FEATURES]
    y = df[TARGET]

    # Held out test rows are never used for training or hyperparameter
    # search, only for the final, honest evaluation at the end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training rows: {len(X_train):,}   Held-out test rows: {len(X_test):,}")

    # The full training set can be millions of rows, which would make grid
    # search very slow. If it is too big, take a smaller random sample
    # (still balanced by class) just for choosing hyperparameters.
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

    # A pipeline bundles the scaler and the model together, so the exact
    # same scaling that is fit here is automatically reused every time this
    # pipeline is used to predict, later on, on new data.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="logistic",
            solver="lbfgs",
            max_iter=300,
            random_state=RANDOM_STATE,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "kappa": make_scorer(cohen_kappa_score)}

    # Grid search tries every combination in PARAM_GRID (5 network sizes x
    # 4 regularisation strengths = 20 combinations), each evaluated with
    # 5 fold cross validation, and keeps whichever combination scored best.
    print("\nRunning grid search (5-fold cross-validation, 20 combinations)...")
    grid_start = time.perf_counter()
    grid = GridSearchCV(pipeline, PARAM_GRID, scoring=scoring, refit="accuracy", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_grid, y_grid)
    grid_time = time.perf_counter() - grid_start
    print(f"Grid search finished in {grid_time:.1f}s ({grid_time / 60:.2f} min)")

    # Save a readable table of every combination tried and how well it did,
    # useful later for plotting or double checking the choice made.
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

    best_size = grid.best_params_["mlp__hidden_layer_sizes"]
    best_decay = grid.best_params_["mlp__alpha"]

    # Now retrain, from scratch, on the FULL training set (not just the
    # grid search sample) using the best hyperparameters just found. This
    # final_model is the one that actually gets saved and used later.
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

    # First honest look at how good the model is, using the test rows it
    # has never seen during training or grid search.
    y_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_kappa = cohen_kappa_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {test_accuracy:.4f}   Kappa: {test_kappa:.4f}")

    # Permutation importance answers "how much worse does the model get if
    # I scramble this one feature and leave everything else alone". A big
    # drop means the model relies heavily on that feature.
    print("\nComputing permutation importance (this re-runs predictions many times)...")
    perm_result = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df["importance_scaled"] = (
        importance_df["importance_mean"] / importance_df["importance_mean"].max() * 100
    )
    importance_path = os.path.join(RESULTS_FOLDER, "ha3_variable_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")

    # Get the model's raw predicted probability of "fit" (class 1) for
    # every test row, needed for the ROC curve and the threshold sweep below.
    fit_class_index = list(final_model.classes_).index(1)
    test_probs = final_model.predict_proba(X_test)[:, fit_class_index]

    # The ROC curve shows the trade off between true positive rate and
    # false positive rate across every possible threshold, all in one plot.
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
    roc_auc = roc_auc_score(y_test, test_probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    roc_df["roc_auc"] = roc_auc
    roc_path = os.path.join(RESULTS_FOLDER, "ha3_roc_curve.csv")
    roc_df.to_csv(roc_path, index=False)
    print(f"Saved: {roc_path}  (ROC AUC = {roc_auc:.4f})")

    # By default a probability of 0.5 or higher counts as "fit", but that
    # is not necessarily the best cutoff for this problem. Try every
    # threshold from 0.01 to 0.99 and record precision, recall, f1 and
    # Youden's J (true positive rate minus false positive rate) for each.
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
        f1_t = (2 * precision_t * recall_t / (precision_t + recall_t)) if (precision_t + recall_t) > 0 else 0.0
        youden_j = tpr_t - fpr_t

        sweep_rows.append([t, tp, fp, tn, fn, precision_t, recall_t, f1_t, tpr_t, fpr_t, youden_j])

    sweep_df = pd.DataFrame(sweep_rows, columns=[
        "threshold", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "tpr", "fpr", "youden_j",
    ])
    sweep_path = os.path.join(RESULTS_FOLDER, "ha3_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved: {sweep_path}")

    # The chosen threshold is whichever one maximised Youden's J, this is
    # a standard way of picking a single "best" cutoff from an ROC style sweep.
    best_row = sweep_df.loc[sweep_df["youden_j"].idxmax()]
    chosen_threshold = float(best_row["threshold"])
    print(f"Chosen threshold (max Youden's J): {chosen_threshold:.2f}  "
          f"(TPR={best_row['tpr']:.3f}, FPR={best_row['fpr']:.3f})")

    # Recompute the final numbers at exactly this chosen threshold, and
    # save them, this is the one and only confusion matrix that matters
    # for how the model will actually behave once deployed.
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

    # Save the trained pipeline (scaler plus network, bundled together) to
    # disk, this is the file that lets anyone reuse the model without
    # retraining anything.
    model_path = "ha3_nn_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved: {model_path}")

    return final_model, chosen_threshold


# ============================================================
#  STAGE 3 - RUN THE PLAIN HEURISTIC (BASELINE)
# ============================================================

def run_baseline_one_order(items):

    """
    Packs a single order using only the plain heuristic (run_ep_method), no model involved,
    and returns the number of bins used plus how much bin space was left unused.
    """

    open_bins, num_bins, cost = run_ep_method(items, bin_types, visualize=False)
    volume_used = sum(
        (x2 - x1) * (y2 - y1) * (z2 - z1)
        for bin_obj in open_bins
        for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
    )
    bin_volume_total = sum(b.length * b.width * b.height for b in open_bins)
    residual_capacity = bin_volume_total - volume_used
    return num_bins, residual_capacity


def run_baseline(dataset):

    """
    Runs the plain heuristic over every order in the whole dataset, one order
    at a time, and records per order results (bins used, cost, space used,
    time taken). This is the baseline everything else gets compared against,
    it never looks at the neural network at all.
    """

    print("\n" + "=" * 60)
    print(" STAGE 3: RUNNING THE PLAIN HEURISTIC (BASELINE)")
    print("=" * 60)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    per_order_rows = []

    start_time = time.perf_counter()
    for idx, order_id in enumerate(order_ids):
        sequence = build_sequence(dataset[order_id])

        # Time just this one order's packing, separate from the rest of
        # the loop, so the per order timing is meaningful.
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
        residual_capacity = bin_volume_total - volume_used

        per_order_rows.append([order_id, len(sequence), num_bins, cost, utilization, order_time, residual_capacity])

        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - start_time
    print(f"Baseline run finished in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    per_order_df = pd.DataFrame(per_order_rows, columns=[
        "order_id", "item_count", "bins_used", "cost", "utilization_pct", "time_seconds", "residual_capacity",
    ])
    save_path = os.path.join(RESULTS_FOLDER, "ha3_baseline_per_order.csv")
    per_order_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return per_order_df



# ============================================================
#  WEIGHT EXTRACTION + RAW NUMPY FORWARD PASS
# ============================================================
#
# Calling sklearn's predict_proba works fine, but it carries some fixed
# overhead every single call (building a DataFrame, sklearn's own internal
# checks). When this needs to run millions of times, once per bin per item,
# that fixed overhead adds up fast. The functions below pull the trained
# network's raw numbers (the scaler's mean and scale, plus each layer's
# weights and biases) straight out of the sklearn pipeline, so predictions
# can be recomputed by hand with plain numpy or even plain python, no
# sklearn call needed at all. This is purely a speed optimisation, the
# actual maths is identical to what sklearn does internally.

def extract_weights(trained_pipeline):

    "Pulls the scaler's mean and scale, plus every layer's weights and biases, "
    "out of a trained sklearn pipeline, so predictions can be recomputed later "
    "without needing sklearn at all."

    scaler = trained_pipeline.named_steps["scaler"]
    mlp = trained_pipeline.named_steps["mlp"]
    scaler_mean = scaler.mean_.astype(np.float64)
    scaler_scale = scaler.scale_.astype(np.float64)
    weights = [w.astype(np.float64) for w in mlp.coefs_]
    biases = [b.astype(np.float64) for b in mlp.intercepts_]
    return scaler_mean, scaler_scale, weights, biases


def save_weights_npz(path, scaler_mean, scaler_scale, weights, biases):
    save_dict = {"scaler_mean": scaler_mean, "scaler_scale": scaler_scale}
    for i, (W, b) in enumerate(zip(weights, biases)):
        save_dict[f"W{i + 1}"] = W
        save_dict[f"b{i + 1}"] = b
    np.savez(path, **save_dict)


def load_weights_npz(path):
    data = np.load(path)
    scaler_mean = data["scaler_mean"]
    scaler_scale = data["scaler_scale"]
    weights, biases = [], []
    i = 1
    while f"W{i}" in data:
        weights.append(data[f"W{i}"])
        biases.append(data[f"b{i}"])
        i += 1
    return scaler_mean, scaler_scale, weights, biases


def sigmoid(z):

    "The logistic activation function, turns any real number into a value between 0 and 1."

    return 1.0 / (1.0 + np.exp(-z))


def nn_forward_batch(X, scaler_mean, scaler_scale, weights, biases):

    """
    X: (n_rows, n_features). Returns (n_rows,) array of P(fit).
    Does exactly what the trained sklearn pipeline does, scale the input,
    then push it through each layer, but using plain numpy matrix math
    instead of calling sklearn, which is much faster for many small calls.
    """
    a = (X - scaler_mean) / scaler_scale
    for W, b in zip(weights, biases):
        a = sigmoid(a @ W + b)
    return a.ravel()


def make_feature_matrix(open_bins, item_l, item_w, item_h):
    """
    Same feature columns as make_feature_row, but written directly into a
    preallocated array instead of building a Python list per bin, calling
    make_feature_row() once per bin, then converting the whole list to a
    numpy array afterwards. Avoids the per-bin function-call overhead and
    the extra list->array copy - matters here since this runs once per
    item, for every open bin, across millions of items.
    """
    n = len(open_bins)
    X = np.empty((n, len(FEATURES)), dtype=np.float64)
    for i, b in enumerate(open_bins):
        X[i, 0] = b.length
        X[i, 1] = b.width
        X[i, 2] = b.height
        X[i, 3] = item_l
        X[i, 4] = item_w
        X[i, 5] = item_h
        X[i, 6] = b.red_cap
        X[i, 7] = b.closest_ep_distance
        X[i, 8] = b.closest_ep_x
        X[i, 9] = b.closest_ep_y
        X[i, 10] = b.closest_ep_z
        X[i, 11] = b.closest_ep_headroom_x
        X[i, 12] = b.closest_ep_headroom_y
        X[i, 13] = b.closest_ep_headroom_z
        X[i, 14] = b.furthest_ep_distance
        X[i, 15] = b.furthest_ep_x
        X[i, 16] = b.furthest_ep_y
        X[i, 17] = b.furthest_ep_z
        X[i, 18] = b.ep_count
    return X


# --- Scalar (pure-Python) forward pass -------------------------------
#
# Numpy has a real, fixed per call cost (array construction, ufunc
# dispatch) that does not shrink no matter how few rows are fed to it.
# For a network this small (9 hidden units or fewer), that fixed cost can
# end up bigger than the actual arithmetic when only a handful of bins
# are open, exactly the small order case where the NN path was still
# slower than the plain heuristic. Below SCALAR_BATCH_THRESHOLD open
# bins, numpy is skipped entirely and the matrix multiply is done by hand
# with plain Python floats instead. Above it, numpy's vectorisation wins
# and the batched path above is used.

SCALAR_BATCH_THRESHOLD = 8


def prepare_scalar_weights(weights_pack):

    """
    Converts the numpy weights_pack into plain Python lists once, up
    front, so the per-item scalar forward pass below never touches numpy
    at all.
    """
    scaler_mean, scaler_scale, weights, biases = weights_pack
    mean_list = scaler_mean.tolist()
    inv_scale_list = (1.0 / scaler_scale).tolist()
    weight_lists = [W.tolist() for W in weights]
    bias_lists = [b.tolist() for b in biases]
    return mean_list, inv_scale_list, weight_lists, bias_lists


def nn_forward_scalar(feature_row, scalar_pack):
    """
    Pure-Python forward pass for ONE bin's feature row (a plain list of
    19 floats). No numpy array creation, no ufunc dispatch, just nested
    loops and math.exp. Returns P(fit) as a plain float.
    """
    mean_list, inv_scale_list, weight_lists, bias_lists = scalar_pack

    # Scale each feature by hand: subtract the mean, multiply by 1/scale
    # (multiplying is a little faster than dividing every time).
    a = [(x - m) * s for x, m, s in zip(feature_row, mean_list, inv_scale_list)]

    # Push the scaled features through each layer of the network by hand.
    for W, b in zip(weight_lists, bias_lists):
        new_a = []
        for j in range(len(b)):
            total = b[j]
            for i in range(len(a)):
                total += a[i] * W[i][j]
            new_a.append(1.0 / (1.0 + math.exp(-total)))
        a = new_a

    return a[0]


def make_feature_list(bin_obj, item_l, item_w, item_h):
    """Same 19 features as make_feature_row, as a plain Python list,
    used by the scalar path so nothing touches numpy."""
    return [
        bin_obj.length, bin_obj.width, bin_obj.height,
        item_l, item_w, item_h,
        bin_obj.red_cap,
        bin_obj.closest_ep_distance, bin_obj.closest_ep_x, bin_obj.closest_ep_y, bin_obj.closest_ep_z,
        bin_obj.closest_ep_headroom_x, bin_obj.closest_ep_headroom_y, bin_obj.closest_ep_headroom_z,
        bin_obj.furthest_ep_distance, bin_obj.furthest_ep_x, bin_obj.furthest_ep_y, bin_obj.furthest_ep_z,
        bin_obj.ep_count,
    ]


def benchmark_inference(pipeline, weights_pack, n_samples=2000):

    """
    Times sklearn's predict_proba against the raw numpy forward pass, at a
    range of batch sizes, and saves the results. This is purely to prove
    the numpy path is actually faster before trusting it for anything else.
    """

    scaler_mean, scaler_scale, weights, biases = weights_pack

    training_data_path = os.path.join(RESULTS_FOLDER, "ha3_training_data.csv")
    if os.path.exists(training_data_path):
        df = pd.read_csv(training_data_path, nrows=5000)
        sample = df[FEATURES].sample(n=min(n_samples, len(df)), random_state=RANDOM_STATE)
        X_full = sample.to_numpy(dtype=np.float64)
    else:
        X_full = np.random.default_rng(RANDOM_STATE).random((n_samples, len(FEATURES)))

    print("\nBenchmarking sklearn predict_proba vs numpy forward pass...")
    rows_out = []
    n_repeats = 200
    fit_class_index = list(pipeline.classes_).index(1)

    for batch_size in (1, 2, 5, 10, 20, 50):
        X = X_full[:batch_size]
        X_df = pd.DataFrame(X, columns=FEATURES)

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            pipeline.predict_proba(X_df)[:, fit_class_index]
        sklearn_time = (time.perf_counter() - t0) / n_repeats

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            nn_forward_batch(X, scaler_mean, scaler_scale, weights, biases)
        numpy_time = (time.perf_counter() - t0) / n_repeats

        speedup = sklearn_time / numpy_time if numpy_time > 0 else float("inf")
        rows_out.append([batch_size, sklearn_time, numpy_time, speedup])
        print(f"  batch_size={batch_size:>3}   sklearn={sklearn_time * 1000:.4f} ms   "
              f"numpy={numpy_time * 1000:.4f} ms   speedup={speedup:.1f}x")

    bench_df = pd.DataFrame(rows_out, columns=[
        "batch_size", "sklearn_predict_proba_seconds", "numpy_forward_pass_seconds", "speedup_factor",
    ])
    bench_path = os.path.join(RESULTS_FOLDER, "ha3_nn_speed_benchmark.csv")
    bench_df.to_csv(bench_path, index=False)
    print(f"Saved: {bench_path}")
    return bench_df


# ============================================================
#  STAGE 4 - RUN THE NN-FILTERED HEURISTIC (FAST / NUMPY PATH)
# ============================================================

def pack_item_nn_fast(open_bins, item_l, item_w, item_h, weights_pack, scalar_pack, threshold, stats, timers):

    """
    Same overall job as pack_item_with_logging, but instead of checking
    every extreme point of every open bin in full, the NN is asked first,
    for each open bin, "does this look like it will fit". Only bins the NN
    approves get the full (slower) extreme-point check. Bins the NN
    rejects are skipped entirely, saving time whenever there are many open
    bins to consider.
    """

    orientations = all_orientations(item_l, item_w, item_h)

    n_open = len(open_bins)

    # With 0 or 1 open bins there is nothing meaningful to choose between,
    # so the NN is skipped entirely and every open bin (there is at most
    # one) is approved automatically.
    if n_open <= 1:
        stats["skipped_single_bin_items"] += 1
        nn_approved = [True] * n_open
    elif n_open <= SCALAR_BATCH_THRESHOLD:
        # Few bins open, use the pure python scalar path, numpy's fixed
        # overhead would cost more here than the arithmetic itself.
        t0 = time.perf_counter()
        nn_approved = []
        for b in open_bins:
            row = make_feature_list(b, item_l, item_w, item_h)
            prob = nn_forward_scalar(row, scalar_pack)
            nn_approved.append(prob >= threshold)
        timers["nn_inference_time"] += time.perf_counter() - t0
        stats["bins_screened"] += n_open
    else:
        # Enough bins open that numpy's vectorisation pays for its own
        # overhead, so all bins are screened together in one batch.
        scaler_mean, scaler_scale, weights, biases = weights_pack
        t0 = time.perf_counter()
        X = make_feature_matrix(open_bins, item_l, item_w, item_h)
        probs = nn_forward_batch(X, scaler_mean, scaler_scale, weights, biases)
        nn_approved = list(probs >= threshold)
        timers["nn_inference_time"] += time.perf_counter() - t0
        stats["bins_screened"] += n_open

    t_geom_start = time.perf_counter()

    # Only bins the NN approved get the full extreme-point check, the
    # actual test of whether the item really fits somewhere inside the bin.
    for bin_obj, approved in zip(open_bins, nn_approved):
        if not approved:
            stats["bins_rejected_by_nn"] += 1
            continue

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
                bin_obj.register_placement(ol, ow, oh)
                placed = True
                break

        if placed:
            timers["geometry_time"] += time.perf_counter() - t_geom_start
            return bin_obj
        else:
            # The NN said yes but the actual extreme-point check said no,
            # keep count of these so it is possible to see how often the
            # NN gets it wrong in this direction.
            if n_open > 1:
                stats["nn_approved_but_geometry_failed"] += 1
            continue

    timers["geometry_time"] += time.perf_counter() - t_geom_start

    # Nothing worked, open a new bin, exactly the same fallback as the
    # plain heuristic uses.
    for bin_type in bin_types:
        for (ol, ow, oh) in orientations:
            if ol <= bin_type["length"] and ow <= bin_type["width"] and oh <= bin_type["height"]:
                new_bin = Bin(bin_type)
                new_bin.placed_items.append((0, 0, 0, ol, ow, oh))
                new_bin.ep_list = update_ep(new_bin.ep_list, 0, 0, 0, ol, ow, oh,
                                            new_bin.placed_items, new_bin.length, new_bin.width, new_bin.height)
                new_bin.register_placement(ol, ow, oh)
                open_bins.append(new_bin)
                return new_bin

    print("Warning: an item could not be packed into any bin type!")
    return None


def run_nn_filtered(dataset, weights_pack, threshold):

    """
    Runs the NN filtered heuristic over the whole dataset, the same as
    run_baseline but using pack_item_nn_fast instead of the plain
    pack_item, and also recording how the NN gatekeeper behaved (how many
    bins it screened, rejected, or got wrong) alongside the usual per
    order packing results.
    """

    print("\n" + "=" * 60)
    print(" STAGE 4: RUNNING THE NN-FILTERED HEURISTIC (FAST PATH)")
    print("=" * 60)
    print(f"Using decision threshold: {threshold:.2f}")

    scalar_pack = prepare_scalar_weights(weights_pack)

    order_ids = list(dataset.keys())
    n_orders = len(order_ids)
    per_order_rows = []
    stats = {
        "bins_screened": 0,
        "bins_rejected_by_nn": 0,
        "nn_approved_but_geometry_failed": 0,
        "skipped_single_bin_items": 0,
    }

    start_time = time.perf_counter()
    for idx, order_id in enumerate(order_ids):
        items = build_sequence(dataset[order_id])
        open_bins = []
        timers = {"nn_inference_time": 0.0, "geometry_time": 0.0}

        order_start = time.perf_counter()
        for item in items:
            pack_item_nn_fast(
                open_bins, item["length"], item["width"], item["height"],
                weights_pack, scalar_pack, threshold, stats, timers,
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
        residual_capacity = bin_volume_total - volume_used
        # Whatever time is not accounted for by NN inference or geometry
        # checks, mostly bookkeeping and loop overhead.
        other_time = order_time - timers["nn_inference_time"] - timers["geometry_time"]

        per_order_rows.append([
            order_id, len(items), num_bins, cost, utilization, order_time, residual_capacity,
            timers["nn_inference_time"], timers["geometry_time"], other_time,
        ])

        if (idx + 1) % 500 == 0 or (idx + 1) == n_orders:
            elapsed = time.perf_counter() - start_time
            print(f"  Processed {idx + 1}/{n_orders} orders ({elapsed:.1f}s elapsed)")

        elapsed = time.perf_counter() - start_time
    print(f"NN-filtered run finished in {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"Bins screened: {stats['bins_screened']:,}   "
          f"Rejected by NN: {stats['bins_rejected_by_nn']:,} "
          f"({stats['bins_rejected_by_nn'] / max(stats['bins_screened'], 1) * 100:.1f}%)   "
          f"NN-approved but geometry failed: {stats['nn_approved_but_geometry_failed']:,}   "
          f"Items skipped (<=1 open bin): {stats['skipped_single_bin_items']:,}")

    per_order_df = pd.DataFrame(per_order_rows, columns=[
        "order_id", "item_count", "bins_used", "cost", "utilization_pct", "time_seconds", "residual_capacity",
        "nn_inference_time_seconds", "geometry_time_seconds", "other_time_seconds",
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
#  STAGE 5 - THRESHOLD SWEEP (FOR THE LINE GRAPH), FAST PATH
# ============================================================

def run_nn_filtered_one_order(items, weights_pack, scalar_pack, threshold):

    "Packs a single order with the NN filtered heuristic at one particular threshold, "
    "and returns bins used plus unused bin space. Used by the sweep below to test "
    "several thresholds quickly on a smaller subset of orders."

    open_bins = []
    stats = {
        "bins_screened": 0, "bins_rejected_by_nn": 0,
        "nn_approved_but_geometry_failed": 0, "skipped_single_bin_items": 0,
    }
    timers = {"nn_inference_time": 0.0, "geometry_time": 0.0}

    for item in items:
        pack_item_nn_fast(
            open_bins, item["length"], item["width"], item["height"],
            weights_pack, scalar_pack, threshold, stats, timers,
        )

    volume_used = sum(
        (x2 - x1) * (y2 - y1) * (z2 - z1)
        for bin_obj in open_bins
        for x1, y1, z1, x2, y2, z2 in bin_obj.placed_items
    )
    bin_volume_total = sum(b.length * b.width * b.height for b in open_bins)
    residual_capacity = bin_volume_total - volume_used
    return len(open_bins), residual_capacity


def run_threshold_sweep(dataset, weights_pack, sweep_thresholds=SWEEP_THRESHOLDS, subset_size=SWEEP_SUBSET_SIZE):

    """
    Runs the baseline and the NN filtered heuristic at several different
    decision thresholds, on a smaller subset of orders (for speed), and
    saves one row per threshold with bins used, unused capacity, and the
    matching precision, recall, f1 and false positive/negative rates from
    the earlier held out test set sweep. This is what analyze3_v2.py later
    turns into the threshold line graph.
    """

    print("\n" + "=" * 60)
    print(" STAGE 5: THRESHOLD SWEEP (FOR THE LINE GRAPH)")
    print("=" * 60)

    order_ids = list(dataset.keys())[:subset_size]
    sequences = [build_sequence(dataset[order_id]) for order_id in order_ids]
    print(f"Using {len(sequences)} orders for the sweep")

    scalar_pack = prepare_scalar_weights(weights_pack)

    full_sweep_path = os.path.join(RESULTS_FOLDER, "ha3_threshold_sweep.csv")
    full_sweep_df = pd.read_csv(full_sweep_path)

    print("Running baseline packing on the subset...")
    base_bins = 0
    base_residual = 0
    for items in sequences:
        num_bins, residual_capacity = run_baseline_one_order(items)
        base_bins += num_bins
        base_residual += residual_capacity

    sweep_rows = []
    for t in sweep_thresholds:
        print(f"Running NN-filtered packing at threshold={t:.2f} ...")

        filt_bins = 0
        filt_residual = 0
        for items in sequences:
            num_bins, residual_capacity = run_nn_filtered_one_order(items, weights_pack, scalar_pack, t)
            filt_bins += num_bins
            filt_residual += residual_capacity

        # Find the row in the big held out test set sweep whose threshold
        # is closest to this one, and borrow its precision/recall/f1/fpr,
        # so the packing results and the model quality metrics line up.
        closest_row = full_sweep_df.iloc[(full_sweep_df["threshold"] - t).abs().idxmin()]
        fnr = 1 - closest_row["tpr"]

        sweep_rows.append([
            t, closest_row["precision"], closest_row["recall"], closest_row["f1"],
            closest_row["fpr"], fnr, filt_bins, filt_residual,
        ])

    sweep_df = pd.DataFrame(sweep_rows, columns=[
        "threshold", "precision", "recall", "f1", "fpr", "fnr", "bins_used", "residual_capacity",
    ])
    sweep_df["baseline_bins"] = base_bins
    sweep_df["baseline_residual_capacity"] = base_residual

    save_path = os.path.join(RESULTS_FOLDER, "ha3_nn_threshold_sweep.csv")
    sweep_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return sweep_df


# ============================================================
#  MAIN
# ============================================================

def main():

    "Runs every stage of the pipeline in order, from raw dataset all the way "
    "to saved results, exactly as described in the header comment at the top of this file."

    script_start = time.perf_counter()
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    dataset = load_dataset(DATA_PATH)

    # Stage 1 and 2, build the training table then train the model on it.
    training_df = generate_training_data(dataset)
    model, chosen_threshold = train_neural_network(training_df)
    # Stage 3, the plain heuristic baseline, no model involved.
    run_baseline(dataset)

    print("\n" + "=" * 60)
    print(" EXTRACTING WEIGHTS FOR THE FAST (NUMPY) PATH")
    print("=" * 60)
    weights_pack = extract_weights(model)
    scaler_mean, scaler_scale, weights, biases = weights_pack

    weights_path = os.path.join(RESULTS_FOLDER, "ha3_nn_weights.npz")
    save_weights_npz(weights_path, scaler_mean, scaler_scale, weights, biases)
    print(f"Saved: {weights_path}")

    # A small table describing the shape of each layer, handy for reports.
    architecture_df = pd.DataFrame([
        {"layer": i + 1, "input_units": W.shape[0], "output_units": W.shape[1], "activation": "logistic"}
        for i, W in enumerate(weights)
    ])
    architecture_path = os.path.join(RESULTS_FOLDER, "ha3_nn_architecture.csv")
    architecture_df.to_csv(architecture_path, index=False)
    print(f"Saved: {architecture_path}")

    # Before trusting the fast numpy path for anything, prove it produces
    # the same probabilities as sklearn's own predict_proba.
    print("\nSanity-checking numpy forward pass against sklearn predict_proba...")
    sample_X = np.random.default_rng(RANDOM_STATE).random((10, len(FEATURES)))
    fit_class_index = list(model.classes_).index(1)
    sklearn_probs = model.predict_proba(pd.DataFrame(sample_X, columns=FEATURES))[:, fit_class_index]
    numpy_probs = nn_forward_batch(sample_X, scaler_mean, scaler_scale, weights, biases)
    max_diff = float(np.max(np.abs(sklearn_probs - numpy_probs)))
    print(f"Max difference between sklearn and numpy probabilities: {max_diff:.2e}")
    if max_diff > 1e-6:
        print("WARNING: numpy forward pass does not closely match sklearn - "
              "double check activation functions / scaler extraction before trusting results below.")

    # Same sanity check again, this time for the pure python scalar path.
    print("Sanity-checking scalar (pure-Python) forward pass against numpy...")
    scalar_pack = prepare_scalar_weights(weights_pack)
    scalar_probs = np.array([nn_forward_scalar(row.tolist(), scalar_pack) for row in sample_X])
    max_diff_scalar = float(np.max(np.abs(numpy_probs - scalar_probs)))
    print(f"Max difference between numpy and scalar probabilities: {max_diff_scalar:.2e}")
    if max_diff_scalar > 1e-6:
        print("WARNING: scalar forward pass does not closely match numpy - "
              "double check nn_forward_scalar before trusting the small-batch path below.")

    benchmark_inference(model, weights_pack)

    # Stage 4 and 5, the NN filtered heuristic over the full dataset, then
    # the threshold sweep on a smaller subset for the line graph.
    run_nn_filtered(dataset, weights_pack, chosen_threshold)
    run_threshold_sweep(dataset, weights_pack)

    total_time = time.perf_counter() - script_start
    print("\n" + "=" * 60)
    print(" ALL STAGES COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"All results saved in the '{RESULTS_FOLDER}' folder.")


if __name__ == "__main__":
    main()