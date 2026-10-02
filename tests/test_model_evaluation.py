"""Tests for SummaryGenerator and ModelEvaluator.

A FakeGenerator replaces the real model, so evaluation, reporting and
promotion logic are tested in milliseconds without downloading anything.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from summarizer.components.model_evaluation import ModelEvaluator
from summarizer.components.model_registry import ModelRegistry
from summarizer.components.summary_generator import SummaryGenerator, postprocess
from summarizer.config import ConfigurationManager
from summarizer.entity.config_entity import GenerationParams
from summarizer.exception import ModelEvaluationError, ModelNotReadyError
from summarizer.pipeline import evaluation_pipeline
from summarizer.utils.common import load_json

TINY = "hf-internal-testing/tiny-random-PegasusForConditionalGeneration"


class FakeGenerator:
    """Returns a canned summary per dialogue (reference = perfect model)."""

    device = "cpu"
    load_seconds = 0.0

    def __init__(self, answers: dict[str, str] | None = None, default: str = "Unrelated text."):
        self.answers = answers or {}
        self.default = default
        self.calls = 0

    def generate(self, texts, params, batch_size=8):
        self.calls += 1
        return [self.answers.get(t, self.default) for t in texts]


@pytest.fixture
def test_rows(tmp_path):
    rows = pd.DataFrame(
        {
            "id": [str(i) for i in range(30)],
            "dialogue": [
                f"Anna: shall we meet at {i}?\nBob: yes, at {i} it is." for i in range(30)
            ],
            "summary": [f"Anna and Bob will meet at {i}." for i in range(30)],
        }
    )
    (tmp_path / "validated").mkdir()
    rows.to_csv(tmp_path / "validated" / "test.csv", index=False)
    return rows


@pytest.fixture
def config(real_config_paths, tmp_path, test_rows):
    cfg = ConfigurationManager(*real_config_paths, smoke_test=True).get_model_evaluation_config()
    reports = tmp_path / "reports"
    return dataclasses.replace(
        cfg,
        validated_dir=tmp_path / "validated",
        reports_dir=reports,
        metrics_file=reports / "evaluation_metrics.json",
        samples_file=reports / "evaluation_samples.csv",
        experiments_file=reports / "generation_experiments.csv",
        history_dir=reports / "evaluations",
        registry_file=tmp_path / "models" / "registry.json",
        num_samples=10,
        auto_promote=True,
    )


@pytest.fixture
def perfect(test_rows):
    return FakeGenerator(dict(zip(test_rows["dialogue"], test_rows["summary"], strict=True)))


def register(config, run_id, smoke=False):
    path = config.registry_file.parent / run_id
    path.mkdir(parents=True, exist_ok=True)
    return ModelRegistry(config.registry_file).register(run_id, path, smoke_test=smoke)


# --------------------------------------------------------- SummaryGenerator
def test_postprocess():
    assert postprocess("  Anna  will call Bob .\n") == "Anna will call Bob."


def test_generator_must_be_loaded_first():
    with pytest.raises(ModelNotReadyError):
        SummaryGenerator("whatever").generate(["text"], GenerationParams())


def test_generator_load_failure_is_model_not_ready(tmp_path):
    with pytest.raises(ModelNotReadyError, match="Could not load model"):
        SummaryGenerator(str(tmp_path / "missing-model")).load()


@pytest.mark.slow
def test_generator_with_tiny_model():
    generator = SummaryGenerator(TINY, device="cpu", max_input_length=512).load()
    assert generator.max_input_length == 200  # capped to the model's position limit
    out = generator.generate(["Anna: hi <file_photo>", "Bob: bye"], GenerationParams(max_length=8))
    assert len(out) == 2 and all(isinstance(s, str) for s in out)
    assert generator.count_tokens("Anna: hi") > 0


# ------------------------------------------------------------ ModelEvaluator
def test_eval_set_is_deterministic(config, perfect):
    evaluator = ModelEvaluator(config, model_ref="some/hub-model", generator=perfect)
    df1, sig1 = evaluator.load_eval_set(10)
    df2, sig2 = evaluator.load_eval_set(10)
    assert df1["id"].tolist() == df2["id"].tolist() and sig1 == sig2
    assert len(evaluator.load_eval_set(1000)[0]) == 30  # capped at the split size


def test_perfect_model_scores_100_and_writes_reports(config, perfect):
    result = ModelEvaluator(config, model_ref="some/hub-model", generator=perfect).run()
    m = result["metrics"]

    assert m["rouge1"] == m["rouge2"] == m["rougeL"] == m["rougeLsum"] == pytest.approx(100.0)
    assert m["bleu"] == pytest.approx(100.0)
    assert m["repeated_trigram_rate"] == 0 and m["empty_outputs"] == 0
    assert m["avg_generated_words"] == m["avg_reference_words"]
    assert result["latency"]["requests"] == config.latency_samples
    assert result["eval_set"]["rows"] == 10
    assert len(result["examples"]) == config.num_examples_to_save
    assert "promotion" not in result  # Hub model: not in the registry

    assert load_json(config.metrics_file)["model"] == "some/hub-model"
    assert (config.history_dir / "some_hub-model__optimized__cpu.json").is_file()
    samples = pd.read_csv(config.samples_file)
    assert {"dialogue", "generated_summary", "reference_summary", "rougeL"} <= set(samples.columns)


def test_unknown_profile(config, perfect):
    with pytest.raises(ModelEvaluationError, match="Unknown generation profile"):
        ModelEvaluator(config, model_ref="m", generator=perfect).run(profile="nope")


def test_no_registered_model_gives_helpful_error(config):
    with pytest.raises(ModelEvaluationError, match="google/pegasus-xsum"):
        ModelEvaluator(config)


def test_missing_test_split(config, perfect, tmp_path):
    (tmp_path / "validated" / "test.csv").unlink()
    with pytest.raises(ModelEvaluationError, match="test split not found"):
        ModelEvaluator(config, model_ref="m", generator=perfect).run()


def test_first_real_model_is_promoted(config, perfect):
    register(config, "run-a")
    result = ModelEvaluator(config, generator=perfect).run()  # default = latest run
    registry = ModelRegistry(config.registry_file)

    assert result["promotion"].startswith("promoted")
    assert registry.get_production()["run_id"] == "run-a"
    assert registry.get("run-a")["metrics"]["rougeL"] == pytest.approx(100.0)


def test_worse_candidate_does_not_replace_production(config, perfect):
    register(config, "good")
    ModelEvaluator(config, model_ref="good", generator=perfect).run()
    register(config, "bad")
    result = ModelEvaluator(config, model_ref="bad", generator=FakeGenerator()).run()

    assert result["promotion"].startswith("kept current production")
    assert ModelRegistry(config.registry_file).get_production()["run_id"] == "good"


def test_better_candidate_replaces_production(config, perfect):
    register(config, "bad")
    ModelEvaluator(config, model_ref="bad", generator=FakeGenerator()).run()
    register(config, "good")
    result = ModelEvaluator(config, model_ref="good", generator=perfect).run()

    registry = ModelRegistry(config.registry_file)
    assert result["promotion"].startswith("promoted: ROUGE-L")
    assert registry.get_production()["run_id"] == "good"
    assert registry.get("bad")["stage"] == "archived"


def test_no_promotion_across_different_eval_sets(config, perfect):
    register(config, "a")
    ModelEvaluator(config, model_ref="a", generator=FakeGenerator()).run()
    register(config, "b")
    other_rows = dataclasses.replace(config, num_samples=20)
    result = ModelEvaluator(other_rows, model_ref="b", generator=perfect).run()
    assert result["promotion"].startswith("skipped: production was evaluated on different rows")


def test_smoke_models_and_disabled_promotion_are_skipped(config, perfect):
    register(config, "smoke", smoke=True)
    assert (
        ModelEvaluator(config, model_ref="smoke", generator=perfect)
        .run()["promotion"]
        .startswith("skipped: smoke-test model")
    )

    register(config, "real")
    result = ModelEvaluator(config, model_ref="real", generator=perfect).run(promote=False)
    assert result["promotion"] == "skipped: auto_promote disabled"
    assert ModelRegistry(config.registry_file).get_production() is None


def test_custom_row_count_is_not_written_to_registry(config, perfect):
    register(config, "run-a")
    result = ModelEvaluator(config, model_ref="run-a", generator=perfect).run(num_samples=5)
    assert "promotion" not in result
    assert ModelRegistry(config.registry_file).get("run-a")["metrics"] == {}
    assert (config.history_dir / "run-a__optimized__cpu__n5.json").is_file()


def test_experiments_table(config, perfect):
    table = ModelEvaluator(config, model_ref="m", generator=perfect).run_experiments()

    assert len(table) == len(config.experiments)
    assert table["experiment"].tolist()[0] == "baseline (greedy)"
    assert table.loc[0, "num_beams"] == 1 and pd.isna(table.loc[0, "length_penalty"])
    assert config.experiments_file.is_file()
    markdown = config.experiments_file.with_suffix(".md").read_text()
    assert "| experiment" in markdown and "model: m" in markdown


def test_cli_returns_1_on_error(monkeypatch):
    def boom(*args, **kwargs):
        raise ModelEvaluationError("no model")

    monkeypatch.setattr(evaluation_pipeline, "ModelEvaluator", boom)
    assert evaluation_pipeline.main([]) == 1
