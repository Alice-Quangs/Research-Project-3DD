# Online 3D bin packing with learned bin filtering

Honours research implementation comparing three constructive heuristics (HA1,
HA2 and HA3), neural-network filtering and XGBoost filtering for online
three-dimensional packing into heterogeneous bins with different costs.

Items are processed sequentially and assigned irrevocably. HA1 and HA2 use
empty maximal spaces; HA3 uses extreme points. Stage 2 uses deterministic
orientation selection: largest feasible base area, then largest base length.
The implemented geometry covers containment and non-overlap. Weight, stability
and blocking constraints are not enforced by this implementation.


## Setup

Run commands from `Main`. The supplied caches and recorded metadata indicate
Python 3.13 was used for saved experiments. The original project declares
Python 3.10+ support; exact saved-model compatibility should be checked using
the recorded environment rather than inferred from that minimum.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Install the supplied pinned environment:

```bash
python -m pip install -r requirements-lock.txt
```

The lock file records the user's environment; these exact versions will need to be installed in the environment. The optional exact solver additionally
requires `python -m pip install ortools`; its not included in the lock file

## Quick checks

```bash
python check_stage2.py
python scripts/audit_submission.py
```

The first command checks orientation, geometry, state preservation and synthetic
stream construction without the dataset or trained models. The second checks
submission structure, recorded model/source hashes and data presence using only
the standard library. It exits nonzero if required inputs or recorded artifacts are missing or changed.

## Data files

The raw dataset and original split are included at these paths. Processed
candidate tables can be regenerated with `generate_dataset.py`:

| Path | Purpose |
|---|---|
| `data/raw/bed-bpp_v1.json` | Raw BED-BPP orders |
| `data/splits/order_splits.json` | Original shared order-level split manifest |
| `data/processed/ha1/`, `ha2/`, `ha3/` | Candidate CSVs and metadata sidecars for training/classification |

Preserve the original split rather than generating a replacement for saved
experiments. The loader sorts numeric `item_sequence` keys; it does not randomly
shuffle arrivals within an order. Synthetic combined streams shuffle source-order
selection and preserve each source order's internal sequence.

## Main entry points

| File or folder | Role |
|---|---|
| `heuristics/` | Active Stage 2 HA1, HA2 and HA3 implementations |
| `config.py`, `bin_types.py` | Shared settings, feature schema and fixed catalogue |
| `data_loader.py`, `splits.py` | Data loading and order-level partitions |
| `features.py`, `filtering.py` | Candidate features and runtime filtering |
| `generate_dataset.py`, `training_data.py` | Candidate generation and training-data validation |
| `neural_network/`, `xgb_model/` | Training and inference adapters |
| `benchmark.py`, `metrics.py`, `plots.py`, `order_size_analysis.py` | Evaluation and reporting |
| `run_stage2.py` | Validation sweep, locked tests and combined workloads |
| `run_combined_timing_stage2.py` | Uninstrumented combined-stream timing |
| `run_combined_diagnostics_stage2.py` | Instrumented geometric/overhead diagnostics |
| `run_clairvoyant_stage2.py`, `clairvoyant_exact.py` | Small-instance full-information benchmark |
| `run_utilisation_diagnostics.py`, `utilisation_reference/` | Utilisation reconciliation and analysis |
| `run_ha*_validation.py`, `run_ha*_test.py` | Earlier benchmark convenience scripts |
| `ha*_variable_class.py` | Stage 2 XGBoost test classification and importance |
| `recover_xgb_outputs.py` | Recovery utility; inspect before use, not part of normal reproduction |
| `models/ha1/`, `ha2/`, `ha3/` | Active trained bundles and training evidence |
| `results/` | Preserved experimental outputs; see index below |
| `archive/models/` | Earlier pre-Stage-2 models, retained for provenance |
| `docs/` | Contains the readme files that contain the step by step walkthrough of the experiments carried out 

`CONFIG.filtering.selected_min_bins` remains `None` as supplied. Stage 2 test
metadata records an activation minimum of **40** for all three heuristics;
pass it explicitly to the Stage 2 runner. This is distinct from each saved
classifier's probability threshold, which is read from model metadata.

The models that are before stage 2 are earlier experiments that did not account for using only the best possible orientation. They are left for reference 
All experiments were carried out using the stage 2 models 

## Reproduction commands

After validating model inference in the recorded environment, a new locked test run is:

```bash
python run_stage2.py --heuristic ha1 --mode test --min-bins 40 --repetitions 7 --output results/reproduction/ha1/test
```

Repeat for HA2 and HA3 with separate output folders. For the paired combined
timing comparison:

```bash
python run_combined_timing_stage2.py --check-only
python run_combined_timing_stage2.py --repetitions 7 --output results/reproduction/combined_timing
```

Use fresh output directories and retain recorded outputs. Do not retrain into
the supplied model directories: trainers reject existing outputs. Classification
scripts can overwrite their own existing outputs; preserve those before rerunning.

Detailed procedures:

- [Stage 2 experiment guide](docs/EXPERIMENT_GUIDE.md) — original workflow; `10` in examples is a placeholder, use the selected `40` for reproduction.
- [Combined timing](docs/README_combined_timing.md)
- [Combined diagnostics](docs/README_combined_diagnostics.md)
- [Clairvoyant comparison](docs/README_clairvoyant.md)
- [Utilisation diagnostics](docs/historical/README.txt)
- [Result index](docs/RESULTS_INDEX.md)

Historical instructions in `docs/historical/README.md` are retained for provenance
and contain superseded setup notes.

