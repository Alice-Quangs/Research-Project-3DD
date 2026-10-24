import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from metrics import classification_metrics, select_threshold

folder = Path("models/ha1/xgb_model")
bundle = joblib.load(folder / "model.joblib")
predictions = pd.read_csv(folder / "validation_predictions.csv")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


metadata = {
    key: value for key, value in bundle.items()
    if key != "model"
}

y = predictions["fit"].to_numpy()
p = predictions["probability"].to_numpy()
threshold = bundle["threshold"]

selection = select_threshold(
    y, p, target_recall=bundle["target_recall"]
)
if not np.isclose(selection["threshold"], threshold):
    raise ValueError(
        "Recovered threshold differs from the saved model. "
        "Check whether config.py has changed."
    )

outputs = {
    "metadata": metadata,
    "validation_metrics": classification_metrics(y, p, threshold),
    "threshold_sweep": selection,
}

for name, content in outputs.items():
    (folder / f"{name}.json").write_text(
        json.dumps(json_safe(content), indent=2, allow_nan=False),
        encoding="utf-8",
    )

print(f"Recovered JSON outputs. Saved threshold: {threshold:.4f}")