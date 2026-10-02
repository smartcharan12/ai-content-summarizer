"""Tests for ConfigurationManager, config entities and runtime settings."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from summarizer.config import AppSettings, ConfigurationManager
from summarizer.constants import PROJECT_ROOT
from summarizer.entity.config_entity import (
    DataValidationConfig,
    GenerationParams,
    TrainingParams,
)
from summarizer.exception import ConfigurationError


# ---------------------------------------------------------------- real files
def test_real_config_builds_every_stage(real_config_paths):
    manager = ConfigurationManager(*real_config_paths, smoke_test=False)

    ingestion = manager.get_data_ingestion_config()
    assert ingestion.hf_dataset_name == "knkarthick/samsum"
    assert ingestion.splits == ("train", "validation", "test")
    assert ingestion.raw_dir.is_absolute()
    assert ingestion.raw_dir.is_relative_to(PROJECT_ROOT)

    validation = manager.get_data_validation_config()
    assert validation.required_columns == ("id", "dialogue", "summary")

    transformation = manager.get_data_transformation_config()
    assert transformation.tokenizer_name == "google/pegasus-xsum"
    assert transformation.max_input_length > transformation.max_target_length

    trainer = manager.get_model_trainer_config()
    assert trainer.base_model == "google/pegasus-xsum"
    assert trainer.training.effective_batch_size == 16
    assert trainer.training.mixed_precision == "auto"

    evaluation = manager.get_model_evaluation_config()
    assert {"baseline", "optimized"} <= set(evaluation.generation_profiles)
    assert evaluation.default_profile == "optimized"


def test_smoke_test_overrides_are_merged(real_config_paths):
    manager = ConfigurationManager(*real_config_paths, smoke_test=True)
    trainer = manager.get_model_trainer_config()

    assert "tiny-random" in trainer.base_model
    assert trainer.training.max_train_samples == 32
    # Keys not mentioned in the smoke block keep their normal values.
    assert trainer.training.learning_rate == pytest.approx(5e-5)
    # The tokenizer must follow the model, or tokens wouldn't match the vocab.
    assert manager.get_data_transformation_config().tokenizer_name == trainer.base_model


def test_smoke_outputs_are_isolated(real_config_paths):
    smoke = ConfigurationManager(*real_config_paths, smoke_test=True)
    real = ConfigurationManager(*real_config_paths, smoke_test=False)
    smoke_root = PROJECT_ROOT / "artifacts" / "smoke"

    for cfg in (smoke.get_model_trainer_config(), smoke.get_model_evaluation_config()):
        assert cfg.models_dir.is_relative_to(smoke_root)
        assert cfg.registry_file.is_relative_to(smoke_root)
    assert smoke.get_data_transformation_config().processed_dir.is_relative_to(smoke_root)
    assert not real.get_model_trainer_config().models_dir.is_relative_to(smoke_root)
    # Model-independent data is shared between modes.
    assert (
        smoke.get_data_validation_config().validated_dir
        == real.get_data_validation_config().validated_dir
    )


def test_smoke_test_flag_read_from_env(real_config_paths, monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "1")
    assert ConfigurationManager(*real_config_paths).smoke_test is True
    monkeypatch.setenv("SMOKE_TEST", "0")
    assert ConfigurationManager(*real_config_paths).smoke_test is False


def test_smoke_block_is_not_exposed_as_a_param(real_config_paths):
    manager = ConfigurationManager(*real_config_paths, smoke_test=False)
    assert "smoke_test" not in manager.params


# ------------------------------------------------------------- broken files
def test_missing_config_file_raises(tmp_path, real_config_paths):
    with pytest.raises(ConfigurationError, match="not found"):
        ConfigurationManager(tmp_path / "nope.yaml", real_config_paths[1])


def test_missing_section_raises_clear_error(write_yaml, real_config_paths):
    config = yaml.safe_load(real_config_paths[0].read_text())
    del config["model_trainer"]
    manager = ConfigurationManager(write_yaml("config.yaml", config), real_config_paths[1])

    with pytest.raises(ConfigurationError, match="model_trainer"):
        manager.get_model_trainer_config()


def test_unknown_training_param_is_rejected(write_yaml, real_config_paths):
    params = yaml.safe_load(real_config_paths[1].read_text())
    params["training"]["learnin_rate"] = 1e-4  # typo
    manager = ConfigurationManager(real_config_paths[0], write_yaml("params.yaml", params))

    with pytest.raises(ConfigurationError, match="Invalid 'training' params"):
        manager.get_model_trainer_config()


def test_unknown_default_generation_profile_is_rejected(write_yaml, real_config_paths):
    params = yaml.safe_load(real_config_paths[1].read_text())
    params["generation"]["default_profile"] = "does-not-exist"
    manager = ConfigurationManager(real_config_paths[0], write_yaml("params.yaml", params))

    with pytest.raises(ConfigurationError, match="default generation profile"):
        manager.get_model_evaluation_config()


# ------------------------------------------------------------- entity rules
def test_generation_params_validation():
    with pytest.raises(ConfigurationError):
        GenerationParams(max_length=10, min_length=20)
    with pytest.raises(ConfigurationError):
        GenerationParams(num_beams=0)


def test_greedy_generation_drops_beam_only_kwargs():
    greedy = GenerationParams(num_beams=1, early_stopping=True).to_dict()
    assert "early_stopping" not in greedy and "length_penalty" not in greedy

    beam = GenerationParams(num_beams=4, early_stopping=True).to_dict()
    assert beam["early_stopping"] is True


def test_validation_config_rejects_inverted_bounds():
    with pytest.raises(ConfigurationError, match="max_dialogue_words"):
        DataValidationConfig(
            raw_dir=Path("r"),
            validated_dir=Path("v"),
            report_file=Path("x.json"),
            splits=("train",),
            required_columns=("id",),
            min_dialogue_words=10,
            max_dialogue_words=5,
            min_summary_words=1,
            drop_duplicates=True,
        )


def test_training_params_require_matching_strategies():
    with pytest.raises(ConfigurationError, match="eval_strategy == save_strategy"):
        TrainingParams(
            num_train_epochs=1,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            weight_decay=0.0,
            warmup_steps=0.0,
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="steps",
            save_total_limit=1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            early_stopping_patience=None,
            mixed_precision="no",
            gradient_checkpointing=False,
        )


# ---------------------------------------------------------- runtime settings
def test_app_settings_read_environment(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_SIZE", "4")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.dev, http://localhost:3000")
    settings = AppSettings(_env_file=None)

    assert settings.max_batch_size == 4
    assert settings.cors_origin_list == ["https://a.dev", "http://localhost:3000"]


def test_app_settings_reject_invalid_values(monkeypatch):
    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(ValueError):
        AppSettings(_env_file=None)


def test_env_example_is_parseable(monkeypatch):
    """.env.example must stay valid: every value in it should load cleanly."""
    for key in AppSettings.model_fields:
        monkeypatch.delenv(key.upper(), raising=False)
    settings = AppSettings(_env_file=PROJECT_ROOT / ".env.example")
    assert settings.model_source == "registry"
    assert settings.cors_origins == "*"
