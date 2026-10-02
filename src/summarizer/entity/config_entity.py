"""Typed configuration objects, one per pipeline stage.

The YAML files are parsed once by :class:`~summarizer.config.configuration.ConfigurationManager`
and turned into these frozen dataclasses. Each stage receives *only* the
settings it needs, with real types (``Path``, ``int``...) instead of a raw dict.
That gives autocomplete, catches typos early, and makes stages easy to unit
test: just build the dataclass by hand.

``__post_init__`` methods reject impossible values (negative lengths, unknown
padding modes) at start-up instead of halfway through a 2-hour training run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from summarizer.exception import ConfigurationError


def _require_positive(name: str, value: int | float) -> None:
    if value is None or value <= 0:
        raise ConfigurationError(f"'{name}' must be > 0", context={name: value})


@dataclass(frozen=True)
class DataIngestionConfig:
    hf_dataset_name: str
    splits: tuple[str, ...]
    raw_dir: Path
    stats_file: Path
    hf_revision: str | None = None


@dataclass(frozen=True)
class DataValidationConfig:
    raw_dir: Path
    validated_dir: Path
    report_file: Path
    splits: tuple[str, ...]
    required_columns: tuple[str, ...]
    min_dialogue_words: int
    max_dialogue_words: int
    min_summary_words: int
    drop_duplicates: bool
    remove_cross_split_leakage: bool = True
    max_drop_fraction: float = 0.05

    def __post_init__(self) -> None:
        _require_positive("min_dialogue_words", self.min_dialogue_words)
        if not 0 <= self.max_drop_fraction < 1:
            raise ConfigurationError(
                "'max_drop_fraction' must be in [0, 1)",
                context={"max_drop_fraction": self.max_drop_fraction},
            )
        _require_positive("min_summary_words", self.min_summary_words)
        if self.max_dialogue_words <= self.min_dialogue_words:
            raise ConfigurationError(
                "'max_dialogue_words' must be greater than 'min_dialogue_words'",
                context={"min": self.min_dialogue_words, "max": self.max_dialogue_words},
            )


@dataclass(frozen=True)
class DataTransformationConfig:
    validated_dir: Path
    processed_dir: Path
    report_file: Path
    splits: tuple[str, ...]
    tokenizer_name: str
    max_input_length: int
    max_target_length: int
    padding: Literal["dynamic", "max_length"]
    normalize_whitespace: bool
    normalize_tags: bool
    lowercase: bool

    def __post_init__(self) -> None:
        _require_positive("max_input_length", self.max_input_length)
        _require_positive("max_target_length", self.max_target_length)
        if self.padding not in ("dynamic", "max_length"):
            raise ConfigurationError(
                "'padding' must be 'dynamic' or 'max_length'", context={"padding": self.padding}
            )


@dataclass(frozen=True)
class TrainingParams:
    """Hyperparameters forwarded to Hugging Face ``Seq2SeqTrainingArguments``."""

    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_steps: float  # int = steps, float in [0, 1) = ratio of total steps
    lr_scheduler_type: str
    eval_strategy: str
    save_strategy: str
    save_total_limit: int
    logging_steps: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    early_stopping_patience: int | None
    mixed_precision: Literal["auto", "bf16", "no"]
    gradient_checkpointing: bool
    optim: str = "adamw_torch"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    keep_checkpoints: bool = False  # project setting, not forwarded to Hugging Face

    def __post_init__(self) -> None:
        _require_positive("num_train_epochs", self.num_train_epochs)
        _require_positive("learning_rate", self.learning_rate)
        _require_positive("per_device_train_batch_size", self.per_device_train_batch_size)
        _require_positive("gradient_accumulation_steps", self.gradient_accumulation_steps)
        if self.mixed_precision not in ("auto", "bf16", "no"):
            raise ConfigurationError(
                "'mixed_precision' must be auto, bf16 or no (fp16 is unstable for PEGASUS)",
                context={"mixed_precision": self.mixed_precision},
            )
        if self.load_best_model_at_end and self.eval_strategy != self.save_strategy:
            raise ConfigurationError(
                "load_best_model_at_end requires eval_strategy == save_strategy",
                context={"eval": self.eval_strategy, "save": self.save_strategy},
            )

    @property
    def effective_batch_size(self) -> int:
        """Samples seen per optimizer step (per device)."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class ModelTrainerConfig:
    processed_dir: Path
    models_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    registry_file: Path
    base_model: str
    seed: int
    max_input_length: int
    max_target_length: int
    smoke_test: bool
    training: TrainingParams


@dataclass(frozen=True)
class GenerationParams:
    """Decoding settings passed to ``model.generate``."""

    max_length: int = 64
    min_length: int = 5
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False

    def __post_init__(self) -> None:
        _require_positive("max_length", self.max_length)
        _require_positive("num_beams", self.num_beams)
        if self.min_length < 0 or self.min_length > self.max_length:
            raise ConfigurationError(
                "'min_length' must be between 0 and 'max_length'",
                context={"min_length": self.min_length, "max_length": self.max_length},
            )

    def to_dict(self) -> dict[str, Any]:
        """Return kwargs for ``model.generate``.

        ``early_stopping`` only has meaning for beam search, so it is dropped
        for greedy decoding (newer ``transformers`` versions warn otherwise).
        """
        kwargs = asdict(self)
        if self.num_beams == 1:
            kwargs.pop("early_stopping")
            kwargs.pop("length_penalty")
        return kwargs


@dataclass(frozen=True)
class ExperimentRun:
    """One decoding configuration in the generation experiments."""

    name: str
    params: GenerationParams


@dataclass(frozen=True)
class ModelEvaluationConfig:
    validated_dir: Path
    models_dir: Path
    reports_dir: Path
    metrics_file: Path
    samples_file: Path
    experiments_file: Path
    history_dir: Path
    registry_file: Path
    max_input_length: int
    num_samples: int
    batch_size: int
    num_examples_to_save: int
    latency_samples: int
    bootstrap_resamples: int
    auto_promote: bool
    seed: int
    smoke_test: bool
    default_profile: str
    generation_profiles: dict[str, GenerationParams] = field(default_factory=dict)
    experiment_samples: int = 100
    experiments: tuple[ExperimentRun, ...] = ()
    # Cleaning must match training (see TextCleaner).
    normalize_whitespace: bool = True
    normalize_tags: bool = True
    lowercase: bool = False

    def __post_init__(self) -> None:
        _require_positive("num_samples", self.num_samples)
        _require_positive("batch_size", self.batch_size)
        _require_positive("experiment_samples", self.experiment_samples)
        if self.latency_samples < 0 or self.bootstrap_resamples < 0:
            raise ConfigurationError("latency_samples and bootstrap_resamples must be >= 0")
        if self.default_profile not in self.generation_profiles:
            raise ConfigurationError(
                "default generation profile is not defined",
                context={
                    "default_profile": self.default_profile,
                    "available": sorted(self.generation_profiles),
                },
            )
