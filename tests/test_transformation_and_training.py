"""Tests for Stage 3 (DataTransformation) and Stage 4 (ModelTrainer).

Tests marked ``slow`` download the ~13 MB tiny random PEGASUS model from the
Hugging Face Hub (cached after the first run) and train it for a few steps on
CPU. Skip them with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from summarizer.components.data_transformation import META_FILE, DataTransformation
from summarizer.components.model_registry import ModelRegistry
from summarizer.components.model_trainer import ModelTrainer, resolve_bf16
from summarizer.config import ConfigurationManager
from summarizer.exception import ConfigurationError, DataTransformationError, ModelTrainingError
from summarizer.utils.common import load_json, save_json

TINY = "hf-internal-testing/tiny-random-PegasusForConditionalGeneration"


def write_validated(directory, n=12):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": str(i),
            "dialogue": f"Anna: meeting at {i} pm? <file_photo>\nBob: yes, see you there {i}",
            "summary": f"Anna and Bob will meet at {i} pm.",
        }
        for i in range(n)
    ]
    for split in ("train", "validation", "test"):
        pd.DataFrame(rows).to_csv(directory / f"{split}.csv", index=False)


@pytest.fixture
def manager(real_config_paths):
    return ConfigurationManager(*real_config_paths, smoke_test=True)


@pytest.fixture
def transform_cfg(manager, tmp_path):
    write_validated(tmp_path / "validated")
    return dataclasses.replace(
        manager.get_data_transformation_config(),
        validated_dir=tmp_path / "validated",
        processed_dir=tmp_path / "processed",
        report_file=tmp_path / "reports" / "transformation_report.json",
    )


@pytest.fixture
def trainer_cfg(manager, tmp_path):
    cfg = manager.get_model_trainer_config()
    return dataclasses.replace(
        cfg,
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        checkpoints_dir=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports",
        registry_file=tmp_path / "models" / "registry.json",
        training=dataclasses.replace(cfg.training, max_train_samples=8, max_eval_samples=4),
    )


# ----------------------------------------------------------- no downloads
def test_missing_validated_split_raises(transform_cfg, tmp_path):
    (tmp_path / "validated" / "test.csv").unlink()
    with pytest.raises(DataTransformationError, match="run the validation stage"):
        DataTransformation(transform_cfg, tokenizer=object()).run()


def test_resolve_bf16_on_cpu():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU-only behaviour")
    assert resolve_bf16("no") is False
    assert resolve_bf16("auto") is False
    with pytest.raises(ConfigurationError, match="bf16"):
        resolve_bf16("bf16")


def test_real_training_params_build_valid_hf_arguments(real_config_paths, tmp_path):
    """The real (non-smoke) params.yaml must be accepted by Seq2SeqTrainingArguments."""
    cfg = ConfigurationManager(*real_config_paths, smoke_test=False).get_model_trainer_config()
    args = ModelTrainer(cfg, registry=ModelRegistry(tmp_path / "r.json")).build_training_args(
        tmp_path / "out"
    )
    assert args.optim == "adafactor"
    assert args.gradient_checkpointing is True
    assert args.eval_strategy == "epoch"
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 16
    assert args.load_best_model_at_end is True
    assert args.report_to in ("none", [], ["none"])


def test_trainer_requires_processed_data(trainer_cfg):
    with pytest.raises(ModelTrainingError, match="run the transformation stage"):
        ModelTrainer(trainer_cfg).check_processed_data()


def test_trainer_rejects_mismatched_tokenizer(trainer_cfg):
    save_json(
        trainer_cfg.processed_dir / META_FILE,
        {"tokenizer_name": "google/pegasus-xsum", "max_input_length": 128, "max_target_length": 64},
    )
    with pytest.raises(ModelTrainingError, match="different settings") as info:
        ModelTrainer(trainer_cfg).check_processed_data()
    assert "tokenizer_name" in info.value.context


def test_resume_needs_run_id(trainer_cfg):
    with pytest.raises(ModelTrainingError, match="run_id"):
        ModelTrainer(trainer_cfg).train(resume=True)


# ------------------------------------------------- tiny model (downloads)
@pytest.mark.slow
def test_transformation_outputs(transform_cfg):
    from datasets import load_from_disk

    report = DataTransformation(transform_cfg).run()
    data = load_from_disk(str(transform_cfg.processed_dir))

    row = data["train"][0]
    assert set(data["train"].column_names) == {"id", "input_ids", "attention_mask", "labels"}
    assert len(row["input_ids"]) == len(row["attention_mask"]) <= transform_cfg.max_input_length
    assert row["labels"][-1] == 1  # </s> (eos) closes every target
    assert report["splits"]["train"]["rows"] == 12
    assert load_json(transform_cfg.processed_dir / META_FILE)["tokenizer_name"] == TINY


@pytest.mark.slow
def test_transformation_cleans_text_before_tokenizing(transform_cfg):
    transformation = DataTransformation(transform_cfg)
    df = transformation.load_and_clean("train")
    assert "[photo]" in df.loc[0, "dialogue"] and "<file_photo>" not in df.loc[0, "dialogue"]


@pytest.mark.slow
def test_max_length_padding_masks_labels(transform_cfg):
    cfg = dataclasses.replace(transform_cfg, padding="max_length", max_target_length=32)
    out = DataTransformation(cfg).tokenize_batch(
        {"dialogue": ["Anna: hi\nBob: hello"], "summary": ["Anna greets Bob."]}
    )
    assert len(out["input_ids"][0]) == cfg.max_input_length
    assert len(out["labels"][0]) == 32
    assert out["labels"][0][-1] == -100  # padding is ignored by the loss
    assert 0 not in out["labels"][0]  # no raw pad ids left in labels


@pytest.mark.slow
def test_train_save_register_and_resume(transform_cfg, trainer_cfg):
    DataTransformation(transform_cfg).run()
    cfg = dataclasses.replace(
        trainer_cfg, training=dataclasses.replace(trainer_cfg.training, keep_checkpoints=True)
    )
    entry = ModelTrainer(cfg).train(run_id="run-1")

    model_dir = cfg.models_dir / "run-1"
    assert entry["stage"] == "candidate" and entry["smoke_test"] is True
    assert (model_dir / "model.safetensors").is_file()
    assert (model_dir / "tokenizer.json").is_file()
    metadata = load_json(model_dir / "metadata.json")
    assert metadata["train_rows"] == 8 and metadata["eval_loss"] > 0
    assert len(metadata["data_fingerprint"]) == 16
    assert (cfg.reports_dir / "training_run-1.json").is_file()
    assert list((cfg.checkpoints_dir / "run-1").glob("checkpoint-*"))

    # Simulate an interrupted run: final model missing, checkpoints present.
    import shutil

    shutil.rmtree(model_dir)
    cfg.registry_file.unlink()
    resumed = ModelTrainer(cfg).train(run_id="run-1", resume=True)
    assert resumed["run_id"] == "run-1"
    assert (model_dir / "model.safetensors").is_file()


@pytest.mark.slow
def test_checkpoints_deleted_by_default(transform_cfg, trainer_cfg):
    DataTransformation(transform_cfg).run()
    ModelTrainer(trainer_cfg).train(run_id="run-2")
    assert not (trainer_cfg.checkpoints_dir / "run-2").exists()
    with pytest.raises(ModelTrainingError, match="already exists"):
        ModelTrainer(trainer_cfg).train(run_id="run-2")
