# %%
"""
XGBoost bin filter for the 3D BPP learnheuristic framework.

Mirrors neural_net.py: same online F53 labelling (median projected util),
but uses compact tabular features instead of a voxel MLP. Plugs into the
same filtered_pack_item path in nn_evaluation.py.
"""

import json
import numpy as np
import joblib

from xgboost import XGBClassifier

import ems_heuristic
import neural_net


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
    bin_vol = float(neural_net.bin_volume(bin_obj))
    used = float(neural_net.bin_used_volume(bin_obj))
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
    Same labelling policy as neural_net.build_training_data:
    among open bins that fit the item, label 1 if projected util >= median.

    Trajectory uses F53 pack_item so state evolution matches the NN filter.
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
                if neural_net.item_fits_in_bin(item_dims, bin_obj):
                    projected_used = (
                        neural_net.bin_used_volume(bin_obj) + (item_l * item_w * item_h)
                    )
                    cost_efficiency = projected_used / neural_net.bin_volume(bin_obj)/bin_obj.cost 
                    candidates.append((bin_obj, cost_efficiency))

            if candidates:
                median_util = float(np.median([u for _, u in candidates]))
                for bin_obj, u in candidates:
                    rows.append(extract_features(bin_obj, item_dims))
                    labels.append(1.0 if u >= median_util else 0.0)

            ems_heuristic.pack_item(bin_types, open_bins, item_l, item_w, item_h)

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

def save_model(model, path="bin_filter_xgb.joblib"):
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


def load_model(path="bin_filter_xgb.joblib"):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


# %%

if __name__ == "__main__":
    import torch
    from online_generator import bin_types

    dataset = torch.load("online_dataset.pt", weights_only=False)
    sequences = dataset["sequences"]

    print("Building XGB training data (subset of sequences)...")
    X, y = build_training_data(sequences, bin_types, max_sequences=20)
    print(f"Generated {len(X)} samples  |  positive rate={y.mean():.3f}" if len(y) else "No samples")

    if len(X):
        model = train_model(X, y)
        save_model(model, "bin_filter_xgb.joblib")

        # Feature importance snapshot
        importances = model.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True)
        print("\nTop features:")
        for name, imp in ranked[:10]:
            print(f"  {name:24s}  {imp:.4f}")
    else:
        print("No training samples generated; check sequences / multi-bin scenarios.")

# %%
