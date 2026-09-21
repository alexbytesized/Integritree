"""Validate draft experiment settings without inventing research parameters."""

import argparse
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from integritree.contracts import Contract, DatasetIdentity, NonemptyText, PositiveInteger
from integritree.settings import load_settings

Seed = Annotated[int, Field(strict=True, ge=0, le=4294967295)]
Ratio = Annotated[float, Field(strict=True, gt=0, le=1)]
FeatureName = Literal[
    "hour_of_day", "day_of_week", "type_CASH_IN", "type_CASH_OUT",
    "type_DEBIT", "type_PAYMENT", "type_TRANSFER", "log_amount",
    "is_zero_amount", "is_merchant_origin", "is_merchant_dest",
]
Stage = Literal["prepare", "train", "evaluate", "explain"]


class SplitConfig(Contract):
    training: Literal[0.8] = 0.8
    validation: Literal[0.1] = 0.1
    testing: Literal[0.1] = 0.1
    stratified: Literal[True] = True


class Seeds(Contract):
    split: Seed | None = None
    model: Seed | None = None
    smote: Seed | None = None


class PreprocessingConfig(Contract):
    features: list[FeatureName] | None = Field(default=None, min_length=1)
    time_convention: NonemptyText | None = None
    normalization: Literal["none", "standard", "minmax"] | None = None
    missing_values: Literal["median_unknown"] | None = None
    duplicates: Literal["drop_exact", "keep"] | None = None

    @model_validator(mode="after")
    def unique_features(self) -> "PreprocessingConfig":
        if self.features is not None and len(self.features) != len(set(self.features)):
            raise ValueError("features must not contain duplicates")
        return self


class RandomForestConfig(Contract):
    n_estimators: PositiveInteger | None = None
    max_depth: PositiveInteger | Literal["unlimited"] | None = None
    min_samples_split: Annotated[int, Field(strict=True, ge=2)] | None = None
    min_samples_leaf: PositiveInteger | None = None
    max_features: Literal["sqrt", "log2", "all"] | None = None
    tuning_procedure: NonemptyText | None = None
    bootstrap: Literal[True] = True
    class_weight: Literal["none"] = "none"


class SmoteConfig(Contract):
    method: Literal["smote_encoded", "smotenc"] | None = None
    sampling_ratio: Ratio | None = None
    k_neighbors: PositiveInteger | None = None


class ScoringConfig(Contract):
    probability_method: Literal["mean_tree_probability", "tree_vote_fraction"] | None = None
    threshold: Annotated[float, Field(strict=True, ge=0, le=1)] | None = None
    tie_policy: Literal["legitimate", "fraud"] | None = None


class EvaluationConfig(Contract):
    pr_auc_method: Literal["average_precision", "trapezoidal"] | None = None
    mcnemar_method: Literal["continuity_corrected", "exact"] | None = None
    discordance_edge_policy: NonemptyText | None = None
    # Preserve the legacy default when reading older configuration snapshots.
    # New experiments explicitly select the approved signed method in YAML.
    percentage_difference: Literal["absolute_over_mean", "signed_over_mean"] = "absolute_over_mean"
    alpha: Literal[0.05] = 0.05


class ShapConfig(Contract):
    output_space: Literal["probability", "raw"] | None = None
    background_strategy: NonemptyText | None = None
    background_size: PositiveInteger | None = None
    explanation_coverage: NonemptyText | None = None


class UnresolvedConfigurationError(ValueError):
    pass


class ExperimentConfig(Contract):
    schema_version: Literal[1] = 1
    experiment_name: NonemptyText
    dataset: DatasetIdentity
    target: Literal["isFraud"] = "isFraud"
    split: SplitConfig = Field(default_factory=SplitConfig)
    seeds: Seeds = Field(default_factory=Seeds)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    smote: SmoteConfig = Field(default_factory=SmoteConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    shap: ShapConfig = Field(default_factory=ShapConfig)

    def unresolved_fields(self, stage: Stage | None = None) -> list[str]:
        groups = {
            "prepare": ["preprocessing", "seeds.split"],
            "train": ["random_forest", "smote", "seeds.model", "seeds.smote",
                      "scoring"],
            "evaluate": ["evaluation"],
            "explain": ["shap"],
        }
        if stage is not None and stage not in groups:
            raise ValueError(f"Unknown stage: {stage}")
        order = list(groups)
        needed = order if stage is None else order[:order.index(stage) + 1]
        paths = [path for name in needed for path in groups[name]]
        values = self.model_dump()
        missing: set[str] = set()

        def visit(value: object, path: str) -> None:
            if value is None:
                missing.add(path)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}.{key}")

        for path in paths:
            value = values
            for part in path.split("."):
                value = value[part]
            visit(value, path)
        return sorted(missing)

    def require_ready(self, stage: Stage) -> None:
        missing = self.unresolved_fields(stage)
        if missing:
            raise UnresolvedConfigurationError(
                f"Configuration is not ready for {stage}; unresolved: " + ", ".join(missing)
            )


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently taking the last value."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError("Experiment YAML mapping keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate experiment YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def load_experiment(path: Path) -> ExperimentConfig:
    try:
        contents = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Cannot read experiment configuration at {path}: {exc}") from exc
    if not isinstance(contents, dict):
        raise ValueError(f"Experiment configuration at {path} must be a YAML mapping")
    try:
        return ExperimentConfig.model_validate(contents)
    except ValueError as exc:
        raise ValueError(f"Invalid experiment configuration at {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path relative to the backend root")
    parser.add_argument("--require-stage", choices=["prepare", "train", "evaluate", "explain"])
    args = parser.parse_args()
    try:
        settings = load_settings()
        path = args.config or settings.experiment_config
        if not path.is_absolute():
            path = settings.backend_root / path
        config = load_experiment(path)
        if args.require_stage:
            config.require_ready(args.require_stage)
    except ValueError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({
        "valid": True,
        "experiment_name": config.experiment_name,
        "unresolved": config.unresolved_fields(args.require_stage),
        "note": "Configuration validation does not execute or approve a research method.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
