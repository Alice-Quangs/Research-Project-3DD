# train_ha2_nn.py
#
# Trains an MLPClassifier (sklearn) on HA2 training data, mirroring the
# 2D dissertation's caret/nnet methodology:
#   - single hidden layer, sigmoid (logistic) activation
#   - grid search over hidden layer size and weight decay (alpha)
#   - 5-fold cross-validation
#   - Accuracy and Cohen's Kappa reported
#   - natural class imbalance is NOT corrected (matches 2D approach)

import time
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from sklearn.inspection import permutation_importance

DATA_PATH = "ha2_training_data.csv"
SAMPLE_SIZE = 300_000       # stratified subsample for tuning (full 20.4M rows is impractical to grid search)
RANDOM_STATE = 42

FEATURES = [
    "bin_length", "bin_width", "bin_height",
    "item_length", "item_width", "item_height",
    "red_cap"
]
TARGET = "fit"

# Grid matching Table 3 in the dissertation (size / decay)
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(1,), (3,), (5,), (7,), (9,)],
    "mlp__alpha": [0.0, 0.0001, 0.001, 0.01],
}


def load_and_sample(path, sample_size, random_state):
    print("Loading full training data...")
    df = pd.read_csv(path)
    print(f"Full dataset: {len(df):,} rows")

    if len(df) > sample_size:
        # Stratified sample - preserves the natural 2.1% / 97.9% class ratio
        df_sample, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[TARGET],
            random_state=random_state
        )
    else:
        df_sample = df

    print(f"Sampled dataset: {len(df_sample):,} rows")
    print(f"Class balance in sample: {(df_sample[TARGET].mean() * 100):.2f}% Fit")

    return df_sample


def main():
    script_start = time.perf_counter()

    df = load_and_sample(DATA_PATH, SAMPLE_SIZE, RANDOM_STATE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="logistic",   # sigmoid, matches nnet's hidden unit activation
            solver="lbfgs",          # closer to nnet's BFGS-based optimizer than adam/sgd
            max_iter=300,            # matches dissertation's max iterations
            random_state=RANDOM_STATE
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nRunning grid search (5-fold CV, 20 combinations)...")
    grid_start = time.perf_counter()
    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    grid_time = time.perf_counter() - grid_start
    print(f"Grid search completed in {grid_time:.1f}s ({grid_time / 60:.2f} min)")

    print("\n" + "=" * 60)
    print(" GRID SEARCH RESULTS")
    print("=" * 60)

    results = pd.DataFrame(grid.cv_results_)
    results["size"] = results["param_mlp__hidden_layer_sizes"].apply(lambda t: t[0])
    results["decay"] = results["param_mlp__alpha"]
    results_table = results[["size", "decay", "mean_test_score"]].rename(
        columns={"mean_test_score": "cv_accuracy"}
    ).sort_values(["size", "decay"])
    print(results_table.to_string(index=False))

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Evaluate best model on held-out test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_kappa = cohen_kappa_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" HELD-OUT TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Kappa: {test_kappa:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Fit", "Fit"]))

    # Variable importance via permutation importance
    # (sklearn's MLP has no built-in importance like nnet/Garson method, so we use
    #  permutation importance on the held-out test set as the closest equivalent)
    print("\nComputing permutation importance...")
    perm_result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std
    }).sort_values("importance_mean", ascending=False)

    # Scale to 0-100 like the dissertation's importance plots
    max_importance = importance_df["importance_mean"].max()
    importance_df["importance_scaled"] = (importance_df["importance_mean"] / max_importance) * 100

    print("\n" + "=" * 60)
    print(" VARIABLE IMPORTANCE (scaled 0-100)")
    print("=" * 60)
    print(importance_df[["feature", "importance_scaled"]].to_string(index=False))

    # Save results
    results_table.to_csv("ha2_nn_grid_results.csv", index=False)
    importance_df.to_csv("ha2_nn_variable_importance.csv", index=False)
    print("\nSaved: ha2_nn_grid_results.csv, ha2_nn_variable_importance.csv")

    # Save the fitted pipeline (StandardScaler + MLP together) for use in
    # NN-filtered packing runs
    joblib.dump(best_model, "ha2_nn_model.joblib")
    print("Saved: ha2_nn_model.joblib")

    total_time = time.perf_counter() - script_start
    print(f"\nTotal script time: {total_time:.1f}s ({total_time / 60:.2f} min)")

    with open("ha2_nn_training_timing.csv", "w", newline="") as f:
        import csv as csv_module
        writer = csv_module.writer(f)
        writer.writerow([
            "sample_size", "best_size", "best_decay", "best_cv_accuracy",
            "test_accuracy", "test_kappa", "grid_search_time_seconds", "total_script_time_seconds"
        ])
        writer.writerow([
            len(df), grid.best_params_["mlp__hidden_layer_sizes"][0], grid.best_params_["mlp__alpha"],
            grid.best_score_, test_accuracy, test_kappa, grid_time, total_time
        ])
    print("Saved: ha2_nn_training_timing.csv")


if __name__ == "__main__":
    main()