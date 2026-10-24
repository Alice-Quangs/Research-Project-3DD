# Online 3D bin packing: shared NN and XGBoost framework

Train six separate classifiers: NN and XGBoost for each of HA1, HA2 and HA3. Both models use the same candidate rows, feature definitions, order partitions and runtime filtering rules for each heuristic.

## 1. Repository layout

```text
config.py
bin_types.py
data_loader.py
features.py
generate_dataset.py
splits.py
filtering.py
benchmark.py
metrics.py
plots.py
training_data.py
heuristics/
    __init__.py
    ha1.py
    ha2.py
    ha3.py
neural_network/
    __init__.py
    train.py
    predictor.py
xgb_model/
    __init__.py
    train.py
    predictor.py
scripts/
    generate_bins.py
    validate_inference.py
data/
    raw/bed-bpp_v1.json
    splits/
    processed/
models/
results/

```

Create empty `__init__.py` files in the three package folders. Keep the two `train.py` and two `predictor.py` files in their respective model folders. Use `xgb_model`, not `xgboost`, to avoid shadowing the installed library. 

Run all commands below from the repository root. Examples use HA1; replace `ha1` with `ha2` or `ha3` throughout to repeat the experiment. Never mix a model from one heuristic with another heuristic's data.

## 2. Environment

Use Python 3.10 or newer. Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install numpy pandas scipy "scikit-learn>=1.7" "xgboost>=2.0" joblib matplotlib
python -m pip freeze > requirements-lock.txt
```

The NN requires `MLPClassifier.fit(sample_weight=...)`. Use the same environment for training and inference, and preserve the lock file with the experiment. The XGBoost reference adapter uses a private API to force the DMatrix path; it explicitly rejects versions where that hook is absent. Validate against your installed version before benchmarking.



## 4. Fixed experiment settings

| Setting | Current implementation |
|---|---|
| Bin costs | 143, 232, 317, 1000 from shared `bin_types.py` |
| Features | 12 common + 7 EMS for HA1/HA2; 12 common + 7 EP for HA3 |
| Splits | 64% train, 16% validation, 20% test; order-level; seed 42 |
| Arrival order | Original numeric dataset sequence, without shuffling items |
| Logging | Every existing open bin before placement; exclude newly opened bin rows |
| Class weights | Negative 1; positive training negatives/training positives |
| Model selection | Highest unweighted validation average precision |
| Threshold | Highest configured grid threshold meeting 99% validation feasible-bin recall |
| Filter-start sweep | 1, 10, 20, 30, 40 open bins; validation only |
| Fallback | Restore all bins only when predicted shortlist is empty |
| Nonempty shortlist fails | Open a new bin without retrying rejected bins |
| Benchmark | Three repeats by default; fresh model/cache each run |

These split and recall settings remain editable defaults. Freeze them before the main experiment. The current arrival policy does not implement a random-arrival experiment. If random sequences are required, that needs a shared saved sequence implementation first.

## 5. Create the saved order split once

Place the source dataset at `data/raw/bed-bpp_v1.json`:

```bash
python splits.py
```

Output: `data/splits/order_splits.json`. Re-running validates and reuses it. A changed dataset or split settings are rejected. All heuristics use this same manifest. Commit it to Git.

## 6. Generate candidate datasets

After completing the heuristic adapters:

```bash
python generate_dataset.py --heuristic ha1 --split train
python generate_dataset.py --heuristic ha1 --split validation
python generate_dataset.py --heuristic ha1 --split test
```

Outputs: `data/processed/ha1/train.csv`, `validation.csv`, `test.csv`, each with a `.meta.json` sidecar. Both model trainers consume the exact same train/validation files. Keep sidecars with their CSVs. Test rows are only for final classification reporting; training scripts do not read them.

Generation is unfiltered: filter-start settings do not require regeneration. It can be expensive because every candidate is labelled. Do not balance or scale these saved files. Each training script handles its own training transformations.

## 7. Train both models

```bash
python -m neural_network.train --heuristic ha1
python -m xgb_model.train --heuristic ha1
```

NN searches 1, 3, 5, 7, 9 hidden units with four alpha values, using logistic activation and L-BFGS. XGBoost searches depths 2, 3, 4, 5 and tree counts 10, 25, 50, 100, 200, with learning rate 0.08. Both grids have 20 candidates, but this does not imply equal computational effort.

To train only the earlier 200-tree/depth-5 XGBoost configuration:

```bash
python -m xgb_model.train --heuristic ha1 --trees 200 --depths 5
```

Choose either the grid or fixed configuration for a given output directory. Existing directories are rejected; use `--output models/ha1/xgb_200` for a separate experiment and update all later model paths accordingly.

Default model paths:

```text
models/ha1/neural_network/model.joblib
models/ha1/xgb_model/model.joblib
```

Each directory also contains metadata, tuning scores, validation predictions and threshold results. XGBoost additionally saves `booster.ubj`. Review convergence warnings in NN `tuning.csv`. Neither trainer refits on validation or uses test data. CSV training data is loaded into RAM.

## 8. Validate inference, then run the validation sweep

The following is Python code. Save it locally as `run_ha1_validation.py` in the repository root, then run `python run_ha1_validation.py`. It reads the actual saved thresholds instead of assuming 0.5.

```python
import json
import subprocess
import sys
from pathlib import Path

h = 'ha1'
xgb = Path('models') / h / 'xgb_model'
nn = Path('models') / h / 'neural_network'
tx = json.loads((xgb / 'metadata.json').read_text())['threshold']
tn = json.loads((nn / 'metadata.json').read_text())['threshold']

for folder, module, threshold, label in [
    (xgb, 'xgb_model.predictor', tx, 'xgb'),
    (nn, 'neural_network.predictor', tn, 'nn'),
]:
    subprocess.run([
        sys.executable, 'scripts/validate_inference.py',
        '--model', str(folder / 'model.joblib'), '--module', module,
        '--rows', f'data/processed/{h}/validation.csv', '--heuristic', h,
        '--threshold', str(threshold),
        '--output', f'results/{h}/{label}_inference.json',
    ], check=True)

subprocess.run([
    sys.executable, 'benchmark.py', '--heuristic', h,
    '--partition', 'validation', '--sweep',
    '--xgb-model', str(xgb / 'model.joblib'),
    '--nn-model', str(nn / 'model.joblib'),
    '--xgb-threshold', str(tx), '--nn-threshold', str(tn),
    '--output', f'results/{h}/validation_sweep',
], check=True)
```

Both inference reports must pass. Default validation samples up to 10,000 rows; `--max-rows 0` checks all rows. These are sampled equivalence checks, not proof for every possible state. Load-time synthetic checks are insufficient alone.

Keep `CONFIG.filtering.selected_min_bins = None` while running `--sweep`; the CLI rejects a sweep when that config value is already set. Reports and result directories must be new on each run.

```bash
python plots.py --results results/ha1/validation_sweep
```

Choose a **common** filter-start value for the controlled NN-versus-XGBoost comparison. Examine runtime together with cost and bin count, and define an acceptable cost change before selecting. The script does not automatically choose a winner. Record the decision and its validation evidence. Do not use test results to select it.

## 9. Run the locked test packing comparison

Save the following as `run_ha1_test.py`. Set `selected_min_bins` to the actual validation choice before running it. The None guard prevents an accidental arbitrary choice.

```python
import json
import subprocess
import sys
from pathlib import Path

h = 'ha1'
selected_min_bins = None  # replace with your validation-selected integer
if selected_min_bins not in (1, 10, 20, 30, 40):
    raise ValueError('Set the filter-start value selected on validation orders.')
xgb = Path('models') / h / 'xgb_model'
nn = Path('models') / h / 'neural_network'
tx = json.loads((xgb / 'metadata.json').read_text())['threshold']
tn = json.loads((nn / 'metadata.json').read_text())['threshold']
subprocess.run([
    sys.executable, 'benchmark.py', '--heuristic', h,
    '--partition', 'test', '--min-bins', str(selected_min_bins),
    '--xgb-model', str(xgb / 'model.joblib'),
    '--nn-model', str(nn / 'model.joblib'),
    '--xgb-threshold', str(tx), '--nn-threshold', str(tn),
    '--output', f'results/{h}/test_comparison',
], check=True)
```

```bash
python run_ha1_test.py
python plots.py --results results/ha1/test_comparison
```

Outputs include per-order CSV, per-run CSV, summary and metadata. Metrics include **number of bins**, cost, volume utilisation, runtime and baseline-relative speedup. Cost change is not an optimality gap. Timing includes features, prediction, geometry and packing updates; excludes loading/reporting. Summary runtime is the median of whole-run totals. Candidate counts are shortlist sizes, not actual geometry-check counts.

## 10. Final classification metrics and variable importance

Packing benchmarking does not automatically compute classification metrics or importance. After settings are locked, save and run this Python code from the root to evaluate identical test rows for both classifiers. Full test classification uses all saved rows. Importance uses one reproducible sample shared across models because it is expensive.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import get_feature_names
from metrics import classification_metrics, permutation_importance
from plots import plot_feature_importance, plot_classification_curves, save_figure
from xgb_model.predictor import load_predictor as load_xgb
from neural_network.predictor import load_predictor as load_nn

h = 'ha1'
frame = pd.read_csv(f'data/processed/{h}/test.csv', dtype={'order_id': str})
X = frame[list(get_feature_names(h))].to_numpy(dtype=float)
y = frame['fit'].to_numpy(dtype=int)
output = Path('results') / h / 'test_classification'
output.mkdir(parents=True, exist_ok=False)
indices = np.sort(np.random.default_rng(42).choice(len(y), min(10000, len(y)), replace=False))
np.save(output / 'importance_row_indices.npy', indices)
probabilities, importance = {}, {}
for name, loader, folder in [
    ('xgboost', load_xgb, 'xgb_model'),
    ('neural_network', load_nn, 'neural_network'),
]:
    predictor = loader(Path('models') / h / folder / 'model.joblib', cache_enabled=False)
    p = predictor.predict_fit_probability(X)
    probabilities[name] = p
    scores = classification_metrics(y, p, predictor.threshold)
    (output / f'{name}_metrics.json').write_text(json.dumps(scores, indent=2))
    result = permutation_importance(predictor, X[indices], y[indices], heuristic=h, repeats=5, seed=42)
    importance[name] = result
    (output / f'{name}_importance.json').write_text(json.dumps(result, indent=2))
fig = plot_feature_importance(importance, title=f'{h.upper()} · test feature importance')
save_figure(fig, output / 'feature_importance')
plt.close(fig)
fig = plot_classification_curves(y, probabilities, title=f'{h.upper()} · test classification')
save_figure(fig, output / 'classification_curves')
plt.close(fig)
```

Importance is the decrease in average precision after permutation; error bars show permutation standard deviation, not confidence intervals. Correlated features may share importance, and shuffling derived features can create physically inconsistent combinations. Preserve negative importances. Do not use final test importance to tune the same experiment. For feature selection, run this analysis on validation rows instead.

Plots use lavender for baseline, violet for XGBoost and deep plum for NN, with PNG and vector PDF exports.

## 11. Optional bin-generation reproduction

```bash
python scripts/generate_bins.py --scope all
```

This reproduces the legacy full-dataset K-means procedure and compares it with the fixed catalogue. It never replaces `bin_types.py`. For a new training-only catalogue experiment, use `--scope train --output results/bins_train`. Changing the catalogue invalidates existing rows and models; regenerate and retrain both models. Full-dataset catalogue derivation must be disclosed when interpreting the held-out evaluation.

## 12. Reproducibility and current validation status

Keep source files, the split manifest, package lock file, chosen settings and compact final summaries under version control. Exclude `.venv/`, `__pycache__/`, large generated CSVs, routine caches and model binaries unless using an appropriate artifact store. Archive trained bundles and their metadata together. Never regenerate the split independently for NN and XGBoost.

The scripts have received targeted synthetic checks. NN fitting and fast/native prediction were smoke-tested. XGBoost training/prediction received syntax checks only because XGBoost was unavailable in the construction environment. The complete pipeline has **not** been run on the real dataset, and the uploaded heuristic adapters still require the changes in section 3. Resolve these steps before full experimental runs or reporting results.
