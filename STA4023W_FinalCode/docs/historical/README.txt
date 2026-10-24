UTILISATION DIAGNOSTICS

Extract this ZIP directly into your existing Main folder. It adds:
  run_utilisation_diagnostics.py
  utilisation_reference/ha1.csv, ha2.csv, ha3.csv
The reference CSVs contain baseline results from repetition 1 of the supplied
stage2/<heuristic>/test/per_order.csv files (2,001 test orders).
No existing project source files need to be replaced.

Activate your usual environment and run from Main:

SMOKE TEST (first 10 saved test orders):
python run_utilisation_diagnostics.py --limit 10 --output results/utilisation_smoke

FULL RUN:
python run_utilisation_diagnostics.py --output results/utilisation_diagnostics

The runner uses config.py, data_loader.py, splits.py, bin_types.py and
heuristics/ha1.py, ha2.py, ha3.py from your existing project. NumPy is required
by those heuristic modules. No ML model or training CSV is loaded.
The default dataset and split paths come from CONFIG. Override with
--dataset PATH and --splits PATH if necessary. Existing splits are validated;
no new split is created. Output must be a new directory to avoid overwriting.
If a run fails, retain its error message and choose a new output path on retry.

OUTPUTS
  overall_summary.csv: volume-weighted utilisation, bin counts, single-item share
  grouped_summary.csv: same quantities by bin type and by single/multiple items
  item_catalogue_summary.csv: item volume relative to smallest feasible bin volume
  ha*_per_order.csv: reconciliation with recorded test run
  ha*_per_bin.csv: final contents count, used/unused capacity, initial utilisation
  ha*_per_item.csv: opening decisions and item/catalogue ratios
  metadata.json: scope, source fingerprints and validation details

All percentage columns are in 0--100 units. Weighted utilisation is total item
volume / total bin volume, matching the thesis, not the unweighted mean of bins.
The smallest feasible bin means minimum VOLUME over types admitting any
orthogonal rotation, not necessarily the first type in catalogue order.
Opening utilisation measures the first item's occupancy of its actual opened bin.
Share of unused volume identifies which groups account for total empty capacity.

The runner checks dimensions, containment, pairwise non-overlap, item counts,
cached volume and per-order cost/bin/volume agreement with the supplied results.
It stops at a mismatch. Weight, stability and blocking are outside these checks.
One deterministic pass per heuristic is used; elapsed times are not benchmarked.

Interpretation: these summaries can reveal catalogue mismatch and where unused
capacity occurs. They do not establish that empty space is unavoidable, quantify
fragmentation separately from online effects, or prove optimal utilisation.

Full data were not supplied, so verification before delivery used synthetic
orders for all three heuristics, deliberately overlapping placements, and a
synthetic full CLI run. Run the smoke test before the complete experiment.

After completion, ZIP results/utilisation_diagnostics and return it for analysis.
