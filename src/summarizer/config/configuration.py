"""ConfigurationManager: the single place that reads YAML config.

Pipeline stages never open YAML files themselves. They ask this class for
their typed config object::

    manager = ConfigurationManager()
    ingestion_cfg = manager.get_data_ingestion_config()

Smoke-test mode
---------------
If the environment variable ``SMOKE_TEST`` is truthy (or ``smoke_test=True``
is passed), the ``smoke_test`` block of ``params.yaml`` is deep-merged over the
normal params: a tiny random model and a few dozen rows. That lets the whole
pipeline run end to end on a CPU laptop or in CI.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from summarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from summarizer.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ExperimentRun,
    GenerationParams,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    TrainingParams,
)
from summarizer.exception import ConfigurationError
from summarizer.logger import get_logger
from summarizer.utils.common import deep_merge, read_yaml, resolve_path

logger = get_logger(__name__)


def _smoke_test_enabled() -> bool:
    return os.getenv("SMOKE_TEST", "").strip().lower() in {"1", "true", "yes", "on"}


class ConfigurationManager:
    """Load ``config.yaml`` + ``params.yaml`` and build per-stage config objects.

    Args:
        config_path: Path to ``config.yaml`` (defaults to ``config/config.yaml``).
        params_path: Path to ``params.yaml`` (defaults to ``config/params.yaml``).
        smoke_test: Force smoke-test mode on/off. ``None`` reads ``SMOKE_TEST``.
    """

    def __init__(
        self,
        config_path: str | Path = CONFIG_FILE_PATH,
        params_path: str | Path = PARAMS_FILE_PATH,
        smoke_test: bool | None = None,
    ) -> None:
        self.config: dict[str, Any] = read_yaml(config_path)
        params = read_yaml(params_path)

        self.smoke_test = _smoke_test_enabled() if smoke_test is None else smoke_test
        smoke_overrides = params.pop("smoke_test", {}) or {}
        if self.smoke_test:
            logger.warning("SMOKE_TEST mode: tiny model + data subset. Metrics are meaningless.")
            params = deep_merge(params, smoke_overrides)
        self.params: dict[str, Any] = params

    # ------------------------------------------------------------------ helpers
    def _section(self, source: dict[str, Any], *keys: str) -> Any:
        """Fetch a nested key, raising a clear error if it is missing."""
        node: Any = source
        for depth, key in enumerate(keys):
            if not isinstance(node, dict) or key not in node:
                missing = ".".join(keys[: depth + 1])
                raise ConfigurationError("Missing configuration key", context={"key": missing})
            node = node[key]
        return node

    def _cfg(self, *keys: str) -> Any:
        return self._section(self.config, *keys)

    def _param(self, *keys: str) -> Any:
        return self._section(self.params, *keys)

    def _output_path(self, value: str) -> Path:
        """Resolve a model-dependent output path (redirected in smoke mode)."""
        path = Path(value)
        if self.smoke_test and not path.is_absolute():
            path = Path(self.config.get("smoke_root", "artifacts/smoke")) / path
        return resolve_path(path)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._cfg("data_ingestion", "splits"))

    @property
    def seed(self) -> int:
        return int(self._param("seed"))

    # ------------------------------------------------------- per-stage configs
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self._cfg("data_ingestion")
        return DataIngestionConfig(
            hf_dataset_name=c["hf_dataset_name"],
            splits=self.splits,
            raw_dir=resolve_path(c["raw_dir"]),
            stats_file=resolve_path(c["stats_file"]),
            hf_revision=c.get("hf_revision"),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        c = self._cfg("data_validation")
        p = self._param("validation")
        try:
            return DataValidationConfig(
                raw_dir=resolve_path(c["raw_dir"]),
                validated_dir=resolve_path(c["validated_dir"]),
                report_file=resolve_path(c["report_file"]),
                splits=self.splits,
                required_columns=tuple(c["required_columns"]),
                min_dialogue_words=int(p["min_dialogue_words"]),
                max_dialogue_words=int(p["max_dialogue_words"]),
                min_summary_words=int(p["min_summary_words"]),
                drop_duplicates=bool(p["drop_duplicates"]),
                remove_cross_split_leakage=bool(p["remove_cross_split_leakage"]),
                max_drop_fraction=float(p["max_drop_fraction"]),
            )
        except KeyError as err:
            raise ConfigurationError(
                "Missing data_validation key", context={"key": str(err)}
            ) from err

    def get_data_transformation_config(self) -> DataTransformationConfig:
        c = self._cfg("data_transformation")
        p = self._param("preprocessing")
        try:
            return DataTransformationConfig(
                validated_dir=resolve_path(c["validated_dir"]),
                processed_dir=self._output_path(c["processed_dir"]),
                report_file=self._output_path(c["report_file"]),
                splits=self.splits,
                tokenizer_name=self._param("model", "base_model"),
                max_input_length=int(p["max_input_length"]),
                max_target_length=int(p["max_target_length"]),
                padding=p["padding"],
                normalize_whitespace=bool(p["normalize_whitespace"]),
                normalize_tags=bool(p["normalize_tags"]),
                lowercase=bool(p["lowercase"]),
            )
        except KeyError as err:
            raise ConfigurationError(
                "Missing preprocessing key", context={"key": str(err)}
            ) from err

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        c = self._cfg("model_trainer")
        try:
            training = TrainingParams(**self._param("training"))
        except TypeError as err:  # unknown or missing hyperparameter name
            raise ConfigurationError(
                "Invalid 'training' params", context={"error": str(err)}
            ) from err
        return ModelTrainerConfig(
            processed_dir=self._output_path(c["processed_dir"]),
            models_dir=self._output_path(c["models_dir"]),
            checkpoints_dir=self._output_path(c["checkpoints_dir"]),
            reports_dir=self._output_path(c["reports_dir"]),
            registry_file=self.registry_file,
            base_model=self._param("model", "base_model"),
            seed=self.seed,
            max_input_length=int(self._param("preprocessing", "max_input_length")),
            max_target_length=int(self._param("preprocessing", "max_target_length")),
            smoke_test=self.smoke_test,
            training=training,
        )

    @property
    def registry_file(self) -> Path:
        return self._output_path(self._cfg("model_registry", "registry_file"))

    def get_generation_profiles(self) -> dict[str, GenerationParams]:
        profiles = self._param("generation", "profiles")
        try:
            return {name: GenerationParams(**values) for name, values in profiles.items()}
        except TypeError as err:
            raise ConfigurationError(
                "Invalid generation profile", context={"error": str(err)}
            ) from err

    def get_experiment_runs(self) -> tuple[ExperimentRun, ...]:
        """Build each experiment = a base profile + the overrides listed for it."""
        exp = self.params.get("experiments") or {}
        profiles = self.get_generation_profiles()
        base_name = exp.get("base_profile", self._param("generation", "default_profile"))
        runs = []
        for spec in exp.get("runs", []):
            spec = dict(spec)
            name = str(spec.pop("name"))
            profile = spec.pop("profile", base_name)
            if profile not in profiles:
                raise ConfigurationError(
                    "Experiment uses an unknown profile", context={"run": name, "profile": profile}
                )
            try:
                params = GenerationParams(**{**asdict(profiles[profile]), **spec})
            except TypeError as err:
                raise ConfigurationError(
                    "Invalid experiment override", context={"run": name, "error": str(err)}
                ) from err
            runs.append(ExperimentRun(name=name, params=params))
        return tuple(runs)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        c = self._cfg("model_evaluation")
        p = self._param("evaluation")
        pre = self._param("preprocessing")
        return ModelEvaluationConfig(
            validated_dir=resolve_path(c["validated_dir"]),
            models_dir=self._output_path(c["models_dir"]),
            reports_dir=self._output_path(c["reports_dir"]),
            metrics_file=self._output_path(c["metrics_file"]),
            samples_file=self._output_path(c["samples_file"]),
            experiments_file=self._output_path(c["experiments_file"]),
            history_dir=self._output_path(c["history_dir"]),
            registry_file=self.registry_file,
            max_input_length=int(pre["max_input_length"]),
            num_samples=int(p["num_samples"]),
            batch_size=int(p["batch_size"]),
            num_examples_to_save=int(p["num_examples_to_save"]),
            latency_samples=int(p["latency_samples"]),
            bootstrap_resamples=int(p["bootstrap_resamples"]),
            auto_promote=bool(p["auto_promote"]),
            seed=self.seed,
            smoke_test=self.smoke_test,
            default_profile=self._param("generation", "default_profile"),
            generation_profiles=self.get_generation_profiles(),
            experiment_samples=int((self.params.get("experiments") or {}).get("num_samples", 100)),
            experiments=self.get_experiment_runs(),
            normalize_whitespace=bool(pre["normalize_whitespace"]),
            normalize_tags=bool(pre["normalize_tags"]),
            lowercase=bool(pre["lowercase"]),
        )
