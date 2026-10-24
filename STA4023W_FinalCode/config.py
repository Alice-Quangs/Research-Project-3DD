"""Shared configuration for the online 3D bin-packing comparison.

Place this file in the repository root. It uses only the standard library.
Importing it does not load data, create folders, train models or run packing.

Example:
    from config import CONFIG, get_feature_names
    columns = get_feature_names("ha1")

The identifiers ha1/ha2 denote the two EMS heuristics; ha3 denotes the
extreme-point heuristic. Adjust these identifiers if the repository uses
different names. Geometry and placement rules belong in heuristics/, not here.

NN training must apply the same positive/negative sample-weight ratio as
XGBoost. Do not replace weighting with undersampling silently.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional, Tuple


# Canonical Python names correct spelling and replace hyphens with underscores.
# Both models must use these names and this exact column order.
COMMON_FEATURES = (
    "item_length",
    "item_width",
    "item_height",
    "item_volume",
    "item_aspect_ratio",
    "bin_length",
    "bin_width",
    "bin_height",
    "residual_volume",
    "current_utilisation",
    "projected_utilisation",
    "item_to_bin_volume_ratio",
)

EMS_FEATURES = (
    "ems_count",
    "max_ems_volume",
    "min_ems_volume",
    "mean_ems_volume",
    "max_ems_length",
    "max_ems_width",
    "max_ems_height",
)

EP_FEATURES = (
    "number_eps",
    "min_eps_z",
    "min_eps_y",
    "min_eps_x",
    "mean_eps_z",
    "mean_eps_y",
    "mean_eps_x",
)

# This version must change whenever feature names, order or definitions change.
FEATURE_SCHEMA_VERSION = "shared_v1"


def get_feature_names(heuristic: str) -> Tuple[str, ...]:
    """Return the 19 columns required by either classifier for a heuristic."""
    if heuristic in ("ha1", "ha2"):
        return COMMON_FEATURES + EMS_FEATURES
    if heuristic == "ha3":
        return COMMON_FEATURES + EP_FEATURES
    raise ValueError(f"Unknown heuristic {heuristic!r}; use ha1, ha2 or ha3.")


@dataclass(frozen=True)
class Paths:
    # Paths resolve relative to this file, independent of the working directory.
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    @property
    def raw_dataset(self) -> Path:
        return self.root / "data" / "raw" / "bed-bpp_v1.json"

    @property
    def split_manifest(self) -> Path:
        return self.root / "data" / "splits" / "order_splits.json"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"


@dataclass(frozen=True)
class DataSettings:
    random_seed: int = 42
    # Proposed defaults: 80% development (64% fit + 16% validation), 20% test.
    # Save exact order IDs once and reuse them across models AND heuristics.
    train_fraction: float = 0.64
    validation_fraction: float = 0.16
    test_fraction: float = 0.20
    split_unit: str = "order"
    # Preserve numeric item_sequence order. Any shuffle experiment must be
    # generated once and replayed identically by all methods.
    arrival_order: str = "dataset_sequence"
    logging_policy: str = "all_open_bins_before_placement"
    include_new_bin_rows: bool = False
    target_column: str = "fit"
    positive_label: int = 1
    negative_label: int = 0
    label_definition: str = "heuristic_finds_feasible_placement"
    metadata_columns: Tuple[str, ...] = ("order_id", "item_index", "bin_index")


@dataclass(frozen=True)
class ClassWeightSettings:
    policy: str = "scale_weighted"
    negative_weight: float = 1.0
    positive_weight_rule: str = "n_negative_divided_by_n_positive"
    undersample: bool = False
    # Calculate from model-fitting labels only (within each training fold
    # during CV). Never use validation/test labels to calculate weights.


@dataclass(frozen=True)
class FilterSettings:
    min_bins_candidates: Tuple[int, ...] = (1, 10, 20, 30, 40)
    # Select separately per heuristic after validation experiments. None is
    # intentional: no winning value has been selected yet.
    selected_min_bins: Optional[int] = None
    activation_rule: str = "open_bin_count_greater_than_or_equal_to_min_bins"
    fallback_policy: str = "restore_all_open_bins_if_shortlist_empty"
    preserve_candidate_order: bool = True
    # Match the uploaded XGBoost behaviour: when a NONEMPTY shortlist has no
    # feasible placement, open a bin; do not retry excluded bins.
    retry_rejected_after_nonempty_shortlist_fails: bool = False


@dataclass(frozen=True)
class ThresholdSettings:
    # Proposed starting settings, editable before retraining. Thresholds are
    # selected independently for each model using the SAME validation rule.
    selection_method: str = "highest_threshold_meeting_recall_target"
    target_recall: float = 0.99
    grid_min: float = 0.0
    grid_max: float = 1.0
    grid_step: float = 0.01
    selection_partition: str = "validation"


@dataclass(frozen=True)
class ModelSettings:
    # Architecture/tuning parameters remain in each model's train.py until
    # agreed. These flags express the common training contract.
    model_names: Tuple[str, ...] = ("xgboost", "neural_network")
    nn_standardise_inputs: bool = True
    xgb_standardise_inputs: bool = False
    nn_weight_argument: str = "sample_weight"
    xgb_weight_argument: str = "scale_pos_weight"
    # Fail explicitly if the chosen NN library cannot accept sample weights.
    require_weighting_support: bool = True


@dataclass(frozen=True)
class BenchmarkSettings:
    partition: str = "test"
    methods: Tuple[str, ...] = ("baseline", "xgboost", "neural_network")
    repetitions: int = 3
    include_feature_extraction_in_runtime: bool = True
    include_placement_and_bin_updates_in_runtime: bool = True
    include_model_loading_in_runtime: bool = False
    # Apply this opportunity to BOTH predictors; validate cached/fast outputs.
    probability_cache_enabled: bool = True
    validate_fast_predictions: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    heuristic: str = "ha1"
    paths: Paths = field(default_factory=Paths)
    data: DataSettings = field(default_factory=DataSettings)
    class_weights: ClassWeightSettings = field(default_factory=ClassWeightSettings)
    filtering: FilterSettings = field(default_factory=FilterSettings)
    thresholds: ThresholdSettings = field(default_factory=ThresholdSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return get_feature_names(self.heuristic)

    def validate(self) -> None:
        get_feature_names(self.heuristic)
        fractions = (self.data.train_fraction, self.data.validation_fraction,
                     self.data.test_fraction)
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("All three split fractions must be between 0 and 1.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Train/validation/test fractions must sum to 1.")
        candidates = self.filtering.min_bins_candidates
        if (not candidates or len(set(candidates)) != len(candidates)
                or any(type(value) is not int or value < 1 for value in candidates)):
            raise ValueError("Filter-start candidates must be unique positive integers.")
        if (self.filtering.selected_min_bins is not None
                and self.filtering.selected_min_bins not in candidates):
            raise ValueError("Selected filter-start value must belong to the sweep.")
        t = self.thresholds
        if not 0 < t.target_recall <= 1:
            raise ValueError("Target recall must be in (0, 1].")
        if not 0 <= t.grid_min < t.grid_max <= 1 or not 0 < t.grid_step <= 1:
            raise ValueError("Invalid probability threshold grid.")
        if self.benchmark.repetitions < 1:
            raise ValueError("Benchmark repetitions must be positive.")
        if self.class_weights.undersample:
            raise ValueError("The agreed weighted policy does not undersample rows.")

    def to_dict(self) -> dict:
        """Return settings suitable for recording alongside every result/model."""
        self.validate()
        settings = asdict(self)
        settings["paths"] = {
            name: str(getattr(self.paths, name))
            for name in ("root", "raw_dataset", "split_manifest", "processed_dir",
                         "models_dir", "results_dir")
        }
        settings["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        settings["feature_names"] = list(self.feature_names)
        return settings


CONFIG = ExperimentConfig()


if __name__ == "__main__":
    # Inspect settings with: python config.py
    # This prints configuration only; it does not execute the experiment.
    print(json.dumps(CONFIG.to_dict(), indent=2))
