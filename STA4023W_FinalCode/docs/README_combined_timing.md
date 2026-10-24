# Uninstrumented combined-order timing comparison



```powershell
python run_combined_timing_stage2.py --check-only
python run_combined_timing_stage2.py --limit-per-target 1 --repetitions 1 --output results/combined_timing_stage2_smoke
python run_combined_timing_stage2.py --repetitions 7 --output results/combined_timing_stage2
```

The same 120 original streams are compared for HA1, HA2 and HA3, with baseline,
NN at 1 and 40 open bins, and XGBoost at 1 and 40 open bins. The runner uses
saved probability thresholds and rotates configuration execution order across
repetitions. Each configuration starts with a freshly loaded predictor; bins are
reset per stream. No model training or threshold tuning occurs.

The default saved cache policy is enabled, matching the uploaded combined
diagnostic results. This is recorded in metadata. Do not change this policy for
the paired follow-up; --cache-policy off exists only for deliberately separate
experiments, which require their own output folder.

Only the full packing loop is timed. No function is wrapped to count geometry
checks or time features, prediction, filtering or packing components. Lightweight
per-item activation and candidate bookkeeping matches the ordinary benchmark
loop. Model loading, progress printing, final validity checking and output occur
outside the measured interval. No individual component times are reported.

Outputs: per_order.csv, per_run.csv, mean_summary.csv, metadata.json and checkpoints.
The summary has target_items=all for whole-run totals plus separate target groups.
runtime_s is the mean whole-run total per repetition, while runtime_s_per_order
is the mean per-stream runtime. Compare methods with the NEW baseline measured
in this runner. Do not mix instrumented and uninstrumented timings.

Use the same command to resume completed runs. Changed parameters, files or
environment require a fresh output folder. The runner rejects mismatched source
heuristics/models and verifies the reconstructed combined-stream fingerprint.

