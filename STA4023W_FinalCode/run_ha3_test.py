import json
import subprocess
import sys
from pathlib import Path

h = 'ha3'
selected_min_bins = 40  # replace with your validation-selected integer
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