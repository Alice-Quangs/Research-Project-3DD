# Larger combined-order diagnostics: Stage 2


## What runs

- The exact 120 previously used combined streams (30 each at targets 100, 200,
  400 and 800), reconstructed from their original source orders, in order.
- Baseline once per repetition, NN at 1 and 40, and XGBoost at 1 and 40.
- HA1, HA2 and HA3; three repetitions by default, rotating method execution order.
- Existing saved models and probability thresholds. No tuning or retraining.
- Existing predictor adapters; cache_enabled is read from the saved combined-run
  configuration (true in the supplied metadata). This is recorded explicitly.
  `--cache-policy off` is available if your actual Stage 2 command explicitly
  disabled caching; use the SAME policy as the earlier run. Do not mix runs with
  different cache settings in one comparison or output folder.

The runner verifies the combined item fingerprint, heuristic-file hashes,
model-file hashes, saved thresholds and Stage 2 orientation probe. It deliberately
does not patch your heuristics. If a file-mismatch check fails, investigate the
changed file rather than bypassing the safeguard.

## Commands (PowerShell, from Main)

Check the sample and all models:

```powershell
python run_combined_diagnostics_stage2.py --check-only
```

Smoke test: one stream from EACH of the four target sizes, all three heuristics:

```powershell
python run_combined_diagnostics_stage2.py --limit-per-target 1 --repetitions 1 --output results/combined_diagnostics_stage2_smoke
```

Full diagnostic experiment:

```powershell
python run_combined_diagnostics_stage2.py --output results/combined_diagnostics_stage2
```

Resume an interrupted run by issuing the same command. Completed whole-method
runs are checkpointed; an interrupted method run restarts. Changed settings,
models, source code, dataset or environment require a fresh output directory.

## Outputs

- `per_order.csv`: per-stream outcomes, counters and timing components.
- `per_run.csv`: total values per method/threshold/heuristic/repetition.
- `mean_summary.csv`: average whole-run totals and separate target-size groups;
  `runtime_s_per_order` is the mean per-stream runtime. `target_items=all` contains
  all 120 streams. Repetitions are not independent new packing instances.
- `metadata.json`: sample, model/source hashes, thresholds, cache flags and timing definitions.
- `checkpoints/`: completed runs for safe resumption.

Use the NEW baseline measured in this experiment for timing comparisons: these
instrumented timings are not interchangeable with the previous seven-repetition
main comparisons. Counting primitive calls adds instrumentation overhead.

`filter_s` INCLUDES `feature_s` and `prediction_s`; do not sum those three.
For a stacked breakdown, use feature_s + prediction_s + pack_call_s + remaining_s.
`remaining_s = runtime_s - feature_s - prediction_s - pack_call_s`.
Packing-call time includes placement search and packing updates, not just geometry.
Geometry counters count `fits_in_ems` for HA1/HA2 and `can_place` for HA3; compare
reductions within each heuristic, not the absolute primitive counts across heuristics.
The counters do not count the final validity checks (these occur outside timing).

This is a diagnostic sensitivity comparison on the existing test streams. It does
not select a new activation threshold or replace the validation-selected main policy.
