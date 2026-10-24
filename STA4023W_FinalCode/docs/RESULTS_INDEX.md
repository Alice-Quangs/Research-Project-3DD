# Recorded results

All paths below are relative to Main. Files were preserved without recalculation.

| Location | Interpretation |
|---|---|
| `results/stage2/ha*/validation/` | Stage 2 validation activation sweep |
| `results/stage2/ha*/test/` | Stage 2 test comparison; metadata records minimum 40 |
| `results/stage2/ha*/combined/` | Stage 2 larger workloads |
| `results/stage2/ha*/baseline/`, `diagnostic1/`, `diagnostic40/` | Additional Stage 2 baseline/diagnostic outputs where present |
| `results/combined_timing_stage2_overnight/` | Seven-repetition uninstrumented combined timing; use when matching the overnight thesis analysis |
| `results/combined_timing_stage2/` | Current extracted summary contains four repetitions per configuration; do not present as a seven-repetition result |
| `results/combined_timing_stage2_before_rep6_rerun/` | Earlier seven-repetition timing evidence, retained separately |
| `results/combined_timing_stage2.zip` | Nonidentical snapshot: differs from the extracted folder; retained intentionally |
| `results/combined_diagnostics_stage2/` | Instrumented component times and geometric counts; not headline runtime |
| `results/clairvoyant_stage2_final/` | Exact/full-information comparison, solver statuses and checkpoints |
| `results/utilisation_diagnostics/` | Detailed utilisation analysis and reconciliation |
| `results/ha1/`, `ha2/`, `ha3/` | Earlier comparisons and classification outputs; inspect child names/metadata to distinguish stages |
| Folders containing `smoke` | Short verification runs, not final experiments |

For Stage 2, `mean_summary.csv` averages repetitions; do not substitute older
`summary.csv` outputs without checking their aggregation. Combined timing uses
`runtime_s` for mean whole-run totals and `runtime_s_per_order` for mean stream
times. Instrumented feature/prediction time is already inside filtering time.

Keep per-order/per-run tables, manifests and checkpoints with their metadata.
Do not pool separate timing runs or silently replace anomalous repetitions.
The overnight summary is identified here as a complete later run.
