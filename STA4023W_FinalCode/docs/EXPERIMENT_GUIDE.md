# Stage 2: orientation correction and filtering experiments

Run commands from your existing Main folder with your existing environment active.
This package targets the shared benchmark.py API and model layout in the saved
project README. It is an addition to that project, not a standalone replacement.

## 1. Preserve the original experiment

Make a copy of Main before installing. Keep the original data, trained models,
results and split manifest. In the working copy, extract this package into Main,
replacing only heuristics/ha1.py, ha2.py and ha3.py. The other files are additions.
Do not use the older uploaded heuristic copies alongside these modules.

Changes: new bins select maximum base area, then maximum x-length using the
existing best_orientation function; orientation deduplication preserves its
original six-orientation enumeration. The latter makes tie behaviour explicit.
First-fit bin selection, EMS/EP scoring, catalogue, original arrival sequences,
empty-shortlist fallback and geometric feasibility scope are preserved.
No weight/stability/blocking constraints are introduced.

Run:

```powershell
python check_stage2.py
```

## 2. Run corrected baselines first

This uses VALIDATION orders and does not need trained models:

```powershell
python run_stage2.py --heuristic ha1 --mode baseline --output results/stage2/ha1/baseline --repetitions 7
python run_stage2.py --heuristic ha2 --mode baseline --output results/stage2/ha2/baseline --repetitions 7
python run_stage2.py --heuristic ha3 --mode baseline --output results/stage2/ha3/baseline --repetitions 7
```

Compare with old-orientation baselines on the SAME validation orders, not the
previous test totals. You can run this baseline command in your preserved old
project using its original heuristic files. The runner does not verify the
orientation version; it hashes the actual heuristic source in metadata.json.
Keep both result folders. Do not choose the orientation policy on test outcomes.

## 3. Regenerate candidate data and retrain

Retain data/splits/order_splits.json exactly. In your working copy, rename the old
data/processed/ha1, ha2, ha3 folders and models/ha1, ha2, ha3 folders to names such
as ha1_before_stage2. Do not delete them. Default trainers refuse existing output
folders. Renaming prevents stale CSVs/models from accidentally being reused.

For EACH h in ha1, ha2 and ha3, run (PowerShell):

```powershell
$h = "ha1"
python generate_dataset.py --heuristic $h --split train
python generate_dataset.py --heuristic $h --split validation
python generate_dataset.py --heuristic $h --split test
python -m neural_network.train --heuristic $h
python -m xgb_model.train --heuristic $h
```

Repeat with $h = "ha2" and "ha3". Stop on any failed command. Use your existing
agreed training options if they differ from the saved README; keep them the same
between stages. These are the existing project trainers, not rewritten trainers.
The runner expects models/<h>/xgb_model/model.joblib and
models/<h>/neural_network/model.joblib, with metadata.json containing threshold.
Use --models-root if your parent model directory differs.

Validate inference with your existing validate_inference.py before benchmarking.
It may be in Main or scripts/ in your local project. For example:

```powershell
$h = "ha1"
$v = "validate_inference.py"
if (Test-Path "scripts/validate_inference.py") { $v = "scripts/validate_inference.py" }
$tx = (Get-Content "models/$h/xgb_model/metadata.json" -Raw | ConvertFrom-Json).threshold
$tn = (Get-Content "models/$h/neural_network/metadata.json" -Raw | ConvertFrom-Json).threshold
New-Item -ItemType Directory -Force "results/stage2/$h" | Out-Null
python $v --model "models/$h/xgb_model/model.joblib" --module xgb_model.predictor --rows "data/processed/$h/validation.csv" --heuristic $h --threshold $tx --output "results/stage2/$h/xgb_inference.json"
python $v --model "models/$h/neural_network/model.joblib" --module neural_network.predictor --rows "data/processed/$h/validation.csv" --heuristic $h --threshold $tn --output "results/stage2/$h/nn_inference.json"
```

Repeat for each heuristic; both reports must pass. If neither validator path
exists, use the current project validator rather than inventing a replacement.

## 4. Sweep activation settings on validation

```powershell
python run_stage2.py --heuristic ha1 --mode validation --output results/stage2/ha1/validation --repetitions 7
python run_stage2.py --heuristic ha2 --mode validation --output results/stage2/ha2/validation --repetitions 7
python run_stage2.py --heuristic ha3 --mode validation --output results/stage2/ha3/validation --repetitions 7
```

Each command evaluates baseline once per repetition and both models at 1, 10,
20, 30, 40 open bins. Unlike the old benchmark CLI --sweep, this calls the API
with explicit settings, so CONFIG.filtering.selected_min_bins need not be None.
Thresholds come from the new metadata, not from the old test results.

Read mean_summary.csv: runtime, speedup (>1 is faster), cost change, bins,
utilisation (fraction), activation %, net candidate reduction %, and fallback %.
Per_order.csv and per_run.csv retain raw observations. These counts are candidate
list counts, not actual geometric work. Choose a COMMON minimum for both models
within each heuristic for the controlled comparison. State your acceptable cost
increase BEFORE selection; do not automatically favour speed if cost deteriorates.
Retain the entire sweep even if every ML setting is slower. Record the choice and
rationale; there is deliberately no automatic winner selector.

## 5. Diagnose the selected setting separately

Set the actual validation choice (10 below is ONLY a syntax example):

```powershell
python run_stage2.py --heuristic ha1 --mode diagnostic --min-bins 10 --output results/stage2/ha1/diagnostic --repetitions 3
```

Repeat for HA2 and HA3 with their choices. diagnostics.csv reports:
- geometry_primitive_calls: fits_in_ems calls for HA1/2, can_place calls for HA3.
  These are orientation-space tests, not actual bin counts or individual EP
  collision comparisons, and are not directly comparable between heuristics.
- feature_s, prediction_s: components already INSIDE filter_s.
- filter_s: filtering including feature/prediction time.
- pack_call_s: geometric search, placement and state/cache updates.
- pack_calls: number of item placements.

Do not sum feature_s + prediction_s + filter_s (double counting). Diagnostic
wrappers add overhead, particularly around cheap primitive calls. Do not use
diagnostic runtimes to claim speedups. Use the uninstrumented validation/test
runs for headline timing. The diagnostic API makes an extra baseline pass before
each learned method; it discards that pass and resets counters before the model
pass. Diagnostic timings are therefore explanatory, not controlled comparisons.

## 6. Locked test comparison

Use each validation-selected minimum, not necessarily the example 10:

```powershell
python run_stage2.py --heuristic ha1 --mode test --min-bins 10 --output results/stage2/ha1/test --repetitions 7
```

Repeat for HA2 and HA3. Each output includes mean_summary.csv and the
mean_order_size_analysis folder with plots and per-order averages. Plot titles
from order_size_analysis do not include heuristic names: identify them by their
parent folder and add titles before thesis use. Packing-means error bars are
one run-level standard deviation; speedup ribbons resample paired orders.
The test set has already been inspected during development; disclose this rather
than describing the new test assessment as completely untouched confirmation.

## 7. Larger-order scalability experiment

```powershell
python run_stage2.py --heuristic ha1 --mode combined --min-bins 10 --sizes 100 200 400 800 --instances 30 --seed 2026 --output results/stage2/ha1/combined --repetitions 7
```

Repeat for HA2/HA3 with the same sizes, instances and seed and their locked minima.
Same source order list + seed gives the same synthetic sequences across methods
and heuristics. The source is test orders only. Whole orders are concatenated,
so actual sizes meet or exceed each target; inspect construction.json for actual
sizes and original IDs. Item order WITHIN each source is preserved. Bins remain
open across source-order boundaries within each synthetic instance. Reset bins
at the next synthetic instance. Items still arrive one at a time; the heuristic
never sees future items. No source order repeats inside one instance, but sources
may be reused across instances/target sizes. These are dependent synthetic
workloads, not 120 independent new real orders. Bootstrap ribbons are exploratory
and do not account for this source reuse. Report original and synthetic findings
separately; do not pool their total costs. Costs are compared only on the same
synthetic instances. Combining assumes items from different orders may share bins.
No training, threshold adjustment or retraining occurs in combined mode.

Keep workload and cache policies fixed. The inherited benchmark loads a fresh
predictor per method/repetition and permits its configured cache to persist
across orders within that run. State this policy, especially for reused source
items. Larger states can be outside the training distribution; record quality as
well as timing. Do not claim a general crossover from one favourable size band.


