# Final-framework clairvoyant benchmark


Use your FINAL Stage 2 `heuristics/ha1.py`, `ha2.py`, and `ha3.py`. The runner
does not change them.

Expected existing inputs:

```
Main/data/raw/bed-bpp_v1.json
Main/models/ha1/xgb_model/model.joblib
Main/models/ha1/neural_network/model.joblib
Main/models/ha2/xgb_model/model.joblib
Main/models/ha2/neural_network/model.joblib
Main/models/ha3/xgb_model/model.joblib
Main/models/ha3/neural_network/model.joblib
```

Activate your existing research environment and open PowerShell in Main.
Install only the additional exact-solver dependency:

```powershell
python -m pip install ortools
```

First validate the instance dimensions, orientation probes, and model loading:

```powershell
python run_clairvoyant_stage2.py --check-only
```

If the orientation probe fails, check that your final Stage 2 heuristics are
installed. The probe checks preferred rotation on one new-bin and one
existing-bin placement; it is a safeguard, not a proof of the entire policy.

Then perform a short end-to-end check (one instance, at most 60 seconds solving):

```powershell
python run_clairvoyant_stage2.py --limit 1 --time-limit 60 --output results/clairvoyant_stage2_smoke
```

Run the full experiment in a different folder:

```powershell
python run_clairvoyant_stage2.py --output results/clairvoyant_stage2_final
```

Defaults: baseline, NN and XGBoost for each of the three heuristics; seven
repetitions; rotating execution order; probability caches disabled; saved model
probability thresholds; filtering activation at 40 open bins. Model loading and
exact solving are excluded from packing runtime. Expect no filtering activations
on these 5–10-item instances, hence equal costs within each heuristic across the
three methods. Packing runtimes are very short and should not carry the main
runtime comparison in the thesis.

The manifest contains the SAME 45 historical prefixes (20 with 5 items, 10 with
6, 5 with 7, 5 with 8, 3 with 9, and 2 with 10), including their dimensions for
verification. These are fixed historical instances; selection was not restricted
to the final ML test partition. Describe this as an exact small-instance benchmark,
not as a held-out classification experiment.

The exact solver uses the current Main bin catalogue and independently considers
all unique orthogonal rotations. It minimises bin cost with containment and
non-overlap. It does NOT enforce weight, support/stability, or blocking. Only use
it as the optimum for this implemented geometric problem; if the final problem
adds other constraints, label it as a relaxed reference instead.

The solver uses the best verified algorithm cost as a feasible upper bound for
safe candidate-bin pruning. It checks packed dimensions, containment and
non-overlap before using these upper bounds. Each instance uses the historical
solver time limit in the manifest. The sum of these limits is the maximum solving
budget, not a runtime prediction. Override with `--time-limit SECONDS` if needed.

## Results and resuming

The output folder contains:

- `exact_results.csv`: one row per instance; solver status, incumbent, best bound,
  solver time and model size. The historical cost is recorded separately.
- `comparison_detailed.csv`: final algorithm costs, mean runtimes and gaps.
- `comparison_summary.csv`: summaries by subset size, heuristic and method.
- `per_run.csv`: individual repeated packing measurements.
- `metadata.json`: dataset, code, instance and model hashes, bin catalogue,
  thresholds, settings and modelling scope.
- `checkpoints/`: one complete result per finished instance.

Repeat the identical full command to resume after interruption. Completed
instances are reused ONLY when settings, input files, source code, model hashes
and environment metadata match. An interrupted in-progress instance is rerun.
Use a new output folder after changing anything; never use the smoke folder for
the full run. Results are exported after each completed instance.

`gap_to_optimum_pct = 100 * (algorithm_cost - optimum) / optimum` is populated
only when SCIP proves OPTIMAL. For a time-limited run, the feasible incumbent is
not called an optimum. `certified_gap_upper_pct` uses the solver's positive lower
bound and bounds the possible relative optimality gap from above; it is not an
exact optimality gap. The older CSV costs do not control the new optimisation.

For a baseline-only run (no model files required), use a separate output folder:

```powershell
python run_clairvoyant_stage2.py --methods baseline --output results/clairvoyant_stage2_baseline
```

