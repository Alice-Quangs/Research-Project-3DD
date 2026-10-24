import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import get_feature_names
from metrics import classification_metrics, permutation_importance
from plots import plot_feature_importance, plot_classification_curves, save_figure
from xgb_model.predictor import load_predictor as load_xgb

h = 'ha3'
frame = pd.read_csv(f'data/processed/{h}/test.csv', dtype={'order_id': str})
X = frame[list(get_feature_names(h))].to_numpy(dtype=float)
y = frame['fit'].to_numpy(dtype=int)
# Use the final Stage 2 test.csv and model.joblib at the paths below.
# Keep these outputs separate from the previous Stage 1 results.
output = Path('results') / h / 'stage2_test_classification_xgboost'
output.mkdir(parents=True, exist_ok=True)
indices = np.sort(np.random.default_rng(42).choice(len(y), min(10000, len(y)), replace=False))
np.save(output / 'importance_row_indices.npy', indices)
probabilities, importance = {}, {}
for name, loader, folder in [
    ('xgboost', load_xgb, 'xgb_model'),
]:
    predictor = loader(Path('models') / h / folder / 'model.joblib', cache_enabled=False)
    p = predictor.predict_fit_probability(X)
    probabilities[name] = p
    scores = classification_metrics(y, p, predictor.threshold)
    (output / f'{name}_metrics.json').write_text(json.dumps(scores, indent=2))
    result = permutation_importance(predictor, X[indices], y[indices], heuristic=h, repeats=5, seed=42)
    importance[name] = result
    (output / f'{name}_importance.json').write_text(json.dumps(result, indent=2))
fig = plot_feature_importance(importance, title=f'{h.upper()} · XGBoost test feature importance')
save_figure(fig, output / f'{h}_xgboost_feature_importance')
plt.close(fig)
fig = plot_classification_curves(y, probabilities, title=f'{h.upper()} · XGBoost test classification')
save_figure(fig, output / f'{h}_xgboost_classification_curves')
plt.close(fig)