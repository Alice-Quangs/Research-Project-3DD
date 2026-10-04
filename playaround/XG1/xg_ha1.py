# %%
"""
XGBoost bin filter for the 3D BPP learnheuristic framework.

Uses compact tabular features, vectorized extraction, physical stability
metrics, and a median-split labeling strategy.
"""

import json
import numpy as np
import joblib

from xgboost import XGBClassifier

import ha1

# %%

FEATURE_NAMES = [
    # item
    "item_l", "item_w", "item_h", "item_volume", "item_max_dim", "item_min_dim", "item_aspect_ratio",
    # bin geometry / cost
    "bin_l", "bin_w", "bin_h", "bin_volume", "bin_cost", "bin_cost_per_volume",
    # fill state
    "used_volume", "residual_volume", "current_util", "projected_util", "item_to_bin_vol_ratio", "num_items",
    # free-space structure (EMS)
    "num_ems", "max_ems_volume", "mean_ems_volume", "max_ems_dim", "supportable_ems_count",
    # free-space structure (Extreme Points) — zeros for EMS bins
    "num_eps", "min_ep_z", "mean_ep_z",
    # 3D Physical Stability
    "support_ratio", "area_fit"
]

# %%
############# STANDALONE GEOMETRY HELPERS #############

def bin_volume(bin_obj):
    return bin_obj.length * bin_obj.width * bin_obj.height

def bin_used_volume(bin_obj):
    total = 0
    for (xs, ys, zs, xe, ye, ze) in bin_obj.placed_items:
        total += (xe - xs) * (ye - ys) * (ze - zs)
    return total

def item_fits_in_bin(item_dims, bin_obj):
    item_l, item_w, item_h = item_dims
    orientations = ha1.all_orientations(item_l, item_w, item_h)
    for ems in bin_obj.ems_list:
        ex, ey, ez, ex_end, ey_end, ez_end = ems
        for (ol, ow, oh) in orientations:
            if ha1.fits_in_ems(ol, ow, oh, ex, ey, ez, ex_end, ey_end, ez_end):
                return True
    return False

# %%
def extract_features_array(open_bins, item_dims):
    """
    Vectorized extraction directly into a NumPy array to bypass dict overhead.
    Adds support_ratio and area_fit physical stability features.
    """
    item_l, item_w, item_h = item_dims
    item_volume = float(item_l * item_w * item_h)
    item_max = float(max(item_l, item_w, item_h))
    item_min = float(min(item_l, item_w, item_h))
    item_aspect_ratio = item_max / item_min if item_min > 0 else 1.0
    item_base_area = float(item_l * item_w)

    n_bins = len(open_bins)
    X = np.zeros((n_bins, len(FEATURE_NAMES)), dtype=np.float32)

    for i, bin_obj in enumerate(open_bins):
        bin_l, bin_w, bin_h = float(bin_obj.length), float(bin_obj.width), float(bin_obj.height)
        bin_vol = bin_l * bin_w * bin_h
        
        used = float(sum((xe - xs) * (ye - ys) * (ze - zs) for (xs, ys, zs, xe, ye, ze) in bin_obj.placed_items))
        residual = max(bin_vol - used, 0.0)
        projected_util = (used + item_volume) / bin_vol if bin_vol > 0 else 0.0
        current_util = used / bin_vol if bin_vol > 0 else 0.0
        
        ems_list = getattr(bin_obj, "ems_list", [])
        num_ems = len(ems_list)
        if num_ems > 0:
            volumes = [(ex_end - ex) * (ey_end - ey) * (ez_end - ez) for (ex, ey, ez, ex_end, ey_end, ez_end) in ems_list]
            max_ems_volume = max(volumes)
            mean_ems_volume = sum(volumes) / num_ems
            max_ems_dim = max(max(ex_end - ex, ey_end - ey, ez_end - ez) for (ex, ey, ez, ex_end, ey_end, ez_end) in ems_list)
            supportable_ems_count = sum(1 for (_, _, ez, _, _, _) in ems_list if ez == 0)
            
            best_ems_base_area = max((ex_end - ex) * (ey_end - ey) for (ex, ey, ez, ex_end, ey_end, ez_end) in ems_list)
            area_fit = item_base_area / best_ems_base_area if best_ems_base_area > 0 else 0.0
        else:
            max_ems_volume = mean_ems_volume = max_ems_dim = supportable_ems_count = area_fit = 0.0

        support_ratio = 1.0 if supportable_ems_count > 0 else 0.5 

        ep_list = getattr(bin_obj, "ep_list", [])
        num_eps = len(ep_list)
        if num_eps > 0:
            zs = [float(z) for (_, _, z) in ep_list]
            min_ep_z, mean_ep_z = min(zs), sum(zs) / num_eps
        else:
            min_ep_z = mean_ep_z = 0.0

        X[i] = [
            item_l, item_w, item_h, item_volume, item_max, item_min, item_aspect_ratio,
            bin_l, bin_w, bin_h, bin_vol, float(bin_obj.cost), float(bin_obj.cost) / bin_vol if bin_vol > 0 else 0.0,
            used, residual, current_util, projected_util, item_volume / bin_vol if bin_vol > 0 else 0.0,
            float(len(bin_obj.placed_items)),
            float(num_ems), float(max_ems_volume), float(mean_ems_volume), float(max_ems_dim), float(supportable_ems_count),
            float(num_eps), float(min_ep_z), float(mean_ep_z),
            float(support_ratio), float(area_fit)
        ]
        
    return X

# %%
def build_training_data(sequences, bin_types, max_sequences=None):
    """
    Uses median split for labeling so the model learns relative ranking,
    but utilizes the fast vectorized feature extraction.
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
                    projected_used = bin_used_volume(bin_obj) + (item_l * item_w * item_h)
                    utilisation = projected_used / bin_volume(bin_obj)
                    candidates.append((bin_obj, utilisation))

            if candidates:
                # Reverting to the median split that preserved packing accuracy
                median_util = float(np.median([u for _, u in candidates]))
                
                candidate_bins = [b for b, _ in candidates]
                features_array = extract_features_array(candidate_bins, item_dims)
                
                for i, (_, u) in enumerate(candidates):
                    rows.append(features_array[i])
                    labels.append(1.0 if u >= median_util else 0.0)

            ha1.pack_item(bin_types, open_bins, item_l, item_w, item_h)

    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.vstack(rows), np.array(labels, dtype=np.float32)

# %%
def train_model(
    X,
    y,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=1.0,
    early_stopping_rounds=20,
    val_fraction=0.15,
    random_state=42,
):
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

def plot_feature_importance(model, top_n=15, save_path="xgb_feature_importance.png", show=True):
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True)[:top_n]
    names, vals = zip(*ranked)

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
def classification_metrics(model, X, y, threshold=0.5):
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

def plot_confusion_matrix(model, X, y, threshold=0.5, save_path="xgb_confusion_matrix.png", show=True):
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

def plot_roc_curve(model, X, y, save_path="xgb_roc_curve.png", show=True):
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
def filter_bins(model, open_bins, item_dims, threshold=0.3):
    """
    Returns the subset of open_bins the XGB model scores as good
    (P(good) >= threshold) for the given item.
    """
    if not open_bins:
        return []

    X = extract_features_array(open_bins, item_dims)
    probs = model.predict_proba(X)[:, 1]
    return [b for b, p in zip(open_bins, probs) if p >= threshold]

# %%
def save_model(model, path="bin_filter_xgb.joblib"):
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(payload, path)
    meta_path = path.replace(".joblib", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURE_NAMES, "n_features": len(FEATURE_NAMES)}, f, indent=2)
    print(f"Model saved to {path}")

def load_model(path="bin_filter_xgb_ha1.joblib"):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload