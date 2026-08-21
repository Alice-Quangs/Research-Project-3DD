# %%
"""
XGBoost bin filter for the 3D BPP learnheuristic framework — F73 (ha2) variant.

Same online median-projected-utilization labelling as xg_ha1.py, but the
trajectory is replayed with ha2.pack_item (F73) instead of ha1.pack_item
(F53), so the bin states / EMS structure candidates trained/labelled here
come from F73's placement behaviour rather than F53's.
"""

import json
import numpy as np
import joblib

from xgboost import XGBClassifier

import ha2


# %%

FEATURE_NAMES = [
    # item
    "item_l",
    "item_w",
    "item_h",
    "item_volume",
    "item_max_dim",
    "item_min_dim",
    "item_aspect_ratio",
    # bin geometry / cost
    "bin_l",
    "bin_w",
    "bin_h",
    "bin_volume",
    "bin_cost",
    "bin_cost_per_volume",
    # fill state
    "used_volume",
    "residual_volume",
    "current_util",
    "projected_util",
    "item_to_bin_vol_ratio",
    "num_items",
    # free-space structure (EMS)
    "num_ems",
    "max_ems_volume",
    "mean_ems_volume",
    "max_ems_dim",
    "supportable_ems_count",
    # free-space structure (Extreme Points) — zeros for EMS bins
    "num_eps",
    "min_ep_z",
    "mean_ep_z",
]


# %%

############# STANDALONE GEOMETRY HELPERS #############
# Helpers for bin and item geometry, free-space structure, and feature extraction.

def bin_volume(bin_obj):
    return bin_obj.length * bin_obj.width * bin_obj.height


def bin_used_volume(bin_obj):
    total = 0
    for (xs, ys, zs, xe, ye, ze) in bin_obj.placed_items:
        total += (xe - xs) * (ye - ys) * (ze - zs)
    return total


def item_fits_in_bin(item_dims, bin_obj):
    item_l, item_w, item_h = item_dims
    orientations = ha2.all_orientations(item_l, item_w, item_h)
    for ems in bin_obj.ems_list:
        ex, ey, ez, ex_end, ey_end, ez_end = ems
        for (ol, ow, oh) in orientations:
            if ha2.fits_in_ems(ol, ow, oh, ex, ey, ez, ex_end, ey_end, ez_end):
                return True
    return False


# %%

def _ems_stats(bin_obj):
    """Summarise Empty Maximal Spaces; EP bins have no ems_list."""
    ems_list = getattr(bin_obj, "ems_list", None)
    if not ems_list:
        return {
            "num_ems": 0.0,
            "max_ems_volume": 0.0,
            "mean_ems_volume": 0.0,
            "max_ems_dim": 0.0,
            "supportable_ems_count": 0.0,
        }

    volumes = []
    max_dim = 0.0
    supportable = 0
    for (ex, ey, ez, ex_end, ey_end, ez_end) in ems_list:
        el = ex_end - ex
        ew = ey_end - ey
        eh = ez_end - ez
        volumes.append(el * ew * eh)
        max_dim = max(max_dim, el, ew, eh)
        # EMS with z == 0 is immediately supportable (EMPTY_SUPPORTED spirit).
        if ez == 0:
            supportable += 1

    return {
        "num_ems": float(len(volumes)),
        "max_ems_volume": float(max(volumes)),
        "mean_ems_volume": float(np.mean(volumes)),
        "max_ems_dim": float(max_dim),
        "supportable_ems_count": float(supportable),
    }


def _ep_stats(bin_obj):
    """Summarise Extreme Points; EMS bins have no ep_list."""
    ep_list = getattr(bin_obj, "ep_list", None)
    if not ep_list:
        return {"num_eps": 0.0, "min_ep_z": 0.0, "mean_ep_z": 0.0}

    zs = [float(z) for (_, _, z) in ep_list]
    return {
        "num_eps": float(len(ep_list)),
        "min_ep_z": float(min(zs)),
        "mean_ep_z": float(np.mean(zs)),
    }


# %%

def extract_features(bin_obj, item_dims):
    """
    Build a fixed-length feature vector for (open bin, arriving item).

    Returns a 1-D np.float32 array aligned with FEATURE_NAMES.
    """
    item_l, item_w, item_h = item_dims
    item_volume = float(item_l * item_w * item_h)
    item_max = float(max(item_l, item_w, item_h))
    item_min = float(min(item_l, item_w, item_h))

    bin_l = float(bin_obj.length)
    bin_w = float(bin_obj.width)
    bin_h = float(bin_obj.height)
    bin_vol = float(bin_volume(bin_obj))
    used = float(bin_used_volume(bin_obj))
    residual = max(bin_vol - used, 0.0)
    projected_util = (used + item_volume) / bin_vol if bin_vol > 0 else 0.0
    current_util = used / bin_vol if bin_vol > 0 else 0.0

    ems = _ems_stats(bin_obj)
    eps = _ep_stats(bin_obj)

    feats = {
        "item_l": float(item_l),
        "item_w": float(item_w),
        "item_h": float(item_h),
        "item_volume": item_volume,
        "item_max_dim": item_max,
        "item_min_dim": item_min,
        "item_aspect_ratio": item_max / item_min if item_min > 0 else 1.0,
        "bin_l": bin_l,
        "bin_w": bin_w,
        "bin_h": bin_h,
        "bin_volume": bin_vol,
        "bin_cost": float(bin_obj.cost),
        "bin_cost_per_volume": float(bin_obj.cost) / bin_vol if bin_vol > 0 else 0.0,
        "used_volume": used,
        "residual_volume": residual,
        "current_util": current_util,
        "projected_util": projected_util,
        "item_to_bin_vol_ratio": item_volume / bin_vol if bin_vol > 0 else 0.0,
        "num_items": float(len(bin_obj.placed_items)),
        **ems,
        **eps,
    }
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


# %%

def build_training_data(sequences, bin_types, max_sequences=None):
    """
    Among open bins that fit the item, label 1 if projected util >= median.

    Trajectory uses F73 (ha2) pack_item, so state evolution matches F73's
    own placement behaviour rather than F53's.
    Returns (X, y) arrays.
    """
    rows = []
    labels = []
    seqs = sequences if max_sequences is None else sequences[:max_sequences]

    for sequence in seqs:
        open_bins = []

        for item in sequence:
            item_l, item_w, item_h = item[0], item[1], item[2]
            item_dims = (item_l, item_w, item_h)

            candidates = []
            for bin_obj in open_bins:
                if item_fits_in_bin(item_dims, bin_obj):
                    projected_used = (
                        bin_used_volume(bin_obj) + (item_l * item_w * item_h)
                    )
                    utilisation = projected_used / bin_volume(bin_obj)
                    candidates.append((bin_obj, utilisation))

            if candidates:
                median_util = float(np.median([u for _, u in candidates]))
                for bin_obj, u in candidates:
                    rows.append(extract_features(bin_obj, item_dims))
                    labels.append(1.0 if u >= median_util else 0.0)

            ha2.pack_item(bin_types, open_bins, item_l, item_w, item_h)

    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.vstack(rows), np.array(labels, dtype=np.float32)


# %%

def train_model(
    X,
    y,
    n_estimators=200,
    max_depth=5,         ## Check if changing these parameters would improve the performance
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=1.0,
    early_stopping_rounds=20,
    val_fraction=0.15,
    random_state=42,
):
    """
    Train an XGBClassifier as a binary bin filter.
    Uses a hold-out slice for early stopping when enough samples exist.
    """
    if len(X) == 0:
        raise ValueError("No training samples provided.")

    n = len(X)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)

    if n >= 40 and val_fraction > 0:
        split = max(1, int(n * (1.0 - val_fraction)))
        train_idx, val_idx = indices[:split], indices[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        X_train, y_train = X[indices], y[indices]
        X_val = y_val = None

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    fit_kwargs = {}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
        # xgboost >= 2.0 uses early_stopping_rounds on the constructor or fit
        try:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
        except Exception:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model.fit(X_train, y_train, **fit_kwargs)

    train_acc = float((model.predict(X_train) == y_train).mean())
    print(f"Train accuracy: {train_acc:.4f}  (n={len(X_train)})")
    if X_val is not None:
        val_acc = float((model.predict(X_val) == y_val).mean())
        print(f"Val accuracy:   {val_acc:.4f}  (n={len(X_val)})")

    return model


def plot_feature_importance(model, top_n=15, save_path="xgb_feature_importance_ha2.png", show=True):
    """
    Plots a horizontal bar chart of the trained model's feature importances
    (as reported by XGBClassifier.feature_importances_) so you can see which
    inputs are driving the "good bin" vs "bad bin" decisions.

    top_n     - how many of the most important features to show
    save_path - where to save the PNG; pass None to skip saving
    show      - whether to open a plot window (set False on headless/servers)

    Returns the ranked (name, importance) pairs that were plotted.
    """
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True)[:top_n]
    names, vals = zip(*ranked)

    # reverse so the most important feature is drawn at the top of the chart
    names = names[::-1]
    vals = vals[::-1]

    plt.figure(figsize=(8, max(4, 0.35 * len(names))))
    plt.barh(names, vals, color="#2b6cb0")
    plt.xlabel("Importance")
    plt.title(f"XGBoost Feature Importance (top {len(names)})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Feature importance plot saved to {save_path}")
    if show:
        plt.show()

    return ranked


# %%

############# CLASSIFIER EVALUATION METRICS #############
# These evaluate the XGB filter itself as a binary classifier (does it
# correctly predict the median-split "good candidate" label), as opposed to
# the downstream packing-outcome metrics (bins/cost/utilization) computed in
# xg_ha2_eval.py. Meant to be run on held-out data, e.g. features/labels
# built from test_sequences via build_training_data.

def classification_metrics(model, X, y, threshold=0.5):
    """
    Returns a dict of accuracy, precision, recall, F1, FPR, FNR, ROC-AUC,
    and raw confusion-matrix counts (tp/tn/fp/fn) for the given features/labels.
    """
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score, f1_score,
        roc_auc_score, accuracy_score,
    )

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    y_int = y.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_int, preds).ravel()

    return {
        "threshold": threshold,
        "n": int(len(y_int)),
        "accuracy": float(accuracy_score(y_int, preds)),
        "precision": float(precision_score(y_int, preds, zero_division=0)),
        "recall": float(recall_score(y_int, preds, zero_division=0)),
        "f1": float(f1_score(y_int, preds, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "roc_auc": float(roc_auc_score(y_int, probs)),
    }


def print_classification_metrics(metrics, label="XGB filter"):
    print(f"\n--- {label}: classifier metrics (threshold={metrics['threshold']}) ---")
    print(f"  n          : {metrics['n']}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  FPR        : {metrics['fpr']:.4f}")
    print(f"  FNR        : {metrics['fnr']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Confusion  : TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")


def plot_confusion_matrix(model, X, y, threshold=0.5,
                           save_path="xgb_confusion_matrix_ha2.png", show=True):
    """Plots and saves a labelled confusion matrix at the given threshold."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y.astype(int), preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bad (0)", "good (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    if show:
        plt.show()

    return cm


def plot_roc_curve(model, X, y, save_path="xgb_roc_curve_ha2.png", show=True):
    """Plots and saves the ROC curve, annotated with AUC."""
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    probs = model.predict_proba(X)[:, 1]
    y_int = y.astype(int)
    fpr, tpr, thresholds = roc_curve(y_int, probs)
    auc = roc_auc_score(y_int, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#2b6cb0", label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to {save_path}")
    if show:
        plt.show()

    return fpr, tpr, auc


# %%

def filter_bins(model, open_bins, item_dims, threshold=0.5):
    """
    Returns the subset of open_bins the XGB model scores as good
    (P(good) >= threshold) for the given item. Same role as neural_net.filter_bins.
    """
    if not open_bins:
        return []

    X = np.vstack([extract_features(b, item_dims) for b in open_bins])
    probs = model.predict_proba(X)[:, 1]
    return [b for b, p in zip(open_bins, probs) if p >= threshold]


# %%

def save_model(model, path="bin_filter_xgb_ha2.joblib"):
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(payload, path)
    # Small companion file for quick inspection without loading the booster
    meta_path = path.replace(".joblib", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURE_NAMES, "n_features": len(FEATURE_NAMES)}, f, indent=2)
    print(f"Model saved to {path}")


def load_model(path="bin_filter_xgb_ha2.joblib"):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


# %%