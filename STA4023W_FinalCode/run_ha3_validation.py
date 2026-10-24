import json
import subprocess
import sys
from pathlib import Path

h = 'ha3'
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