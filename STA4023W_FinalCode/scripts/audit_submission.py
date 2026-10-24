"""Read-only submission checks. Run from any directory; uses standard library only."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    problems = []
    sources = list(ROOT.glob('*.py'))
    for folder in ('heuristics', 'neural_network', 'xgb_model', 'scripts'):
        sources.extend((ROOT / folder).glob('*.py'))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            problems.append(f'{path.relative_to(ROOT)}: {exc}')
    print(f'Checked Python syntax: {len(sources)} source files.')

    manifest = json.loads((ROOT / 'combined_diagnostics_manifest.json').read_text())
    matched = 0
    for heuristic, expected in manifest['heuristics'].items():
        checks = [(ROOT / 'heuristics' / f'{heuristic}.py', expected['heuristic_sha256'])]
        for method, details in expected['models'].items():
            folder = 'xgb_model' if method == 'xgboost' else 'neural_network'
            base = ROOT / 'models' / heuristic / folder
            checks.append((base / 'model.joblib', details['sha256']))
            metadata = json.loads((base / 'metadata.json').read_text())
            if metadata['heuristic'] != heuristic or metadata['threshold'] != details['threshold']:
                problems.append(f'{heuristic}/{method}: metadata disagrees with combined manifest.')
        for path, expected_hash in checks:
            if not path.is_file() or sha256(path) != expected_hash:
                problems.append(f'Missing or changed artifact: {path.relative_to(ROOT)}')
            else:
                matched += 1
    print(f'Matching source/model hashes: {matched}/9.')

    for relative in ('data/raw/bed-bpp_v1.json', 'data/splits/order_splits.json'):
        if not (ROOT / relative).is_file():
            problems.append(f'Required reproduction input missing: {relative}')
    for heuristic in ('ha1', 'ha2', 'ha3'):
        for partition in ('train', 'validation', 'test'):
            path = ROOT / 'data' / 'processed' / heuristic / f'{partition}.csv'
            if not path.is_file():
                print(f'Optional for packing-only replay; needed for training/classification: {path.relative_to(ROOT)} missing.')
    print('This audit does not load models, validate dataset fingerprints, or rerun experiments.')
    for problem in problems:
        print('ISSUE:', problem)
    if not problems:
        print('Submission structure checks passed.')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
