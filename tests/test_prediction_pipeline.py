"""Tests for the PredictionPipeline (input validation, metadata, model selection)."""

from __future__ import annotations

import pytest

from summarizer.components.model_registry import ModelRegistry
from summarizer.config import AppSettings, ConfigurationManager
from summarizer.exception import ConfigurationError, InputValidationError, ModelNotReadyError
from summarizer.pipeline.prediction_pipeline import PredictionPipeline

DIALOGUE = "Anna: Are we still meeting at noon?\nBob: Yes, see you at the cafe."
TINY = "hf-internal-testing/tiny-random-PegasusForConditionalGeneration"


class FakeGenerator:
    device = "cpu"
    load_seconds = 0.01
    max_input_length = 512

    def __init__(self, summary="Anna and Bob meet at noon.", tokens=20, loaded=True):
        self.summary, self.tokens, self.is_loaded = summary, tokens, loaded
        self.batches: list[list[str]] = []

    def load(self):
        self.is_loaded = True
        return self

    def count_tokens(self, text):
        return self.tokens

    def generate(self, texts, params, batch_size=8):
        self.batches.append(list(texts))
        return [self.summary] * len(texts)


def settings(**overrides) -> AppSettings:
    values = {"model_source": "path", "model_name_or_path": "some/model", **overrides}
    return AppSettings(_env_file=None, **values)


@pytest.fixture
def config(real_config_paths, tmp_path):
    manager = ConfigurationManager(*real_config_paths, smoke_test=False)
    manager.config["model_registry"]["registry_file"] = str(tmp_path / "models" / "registry.json")
    return manager


@pytest.fixture
def pipeline(config):
    return PredictionPipeline(settings(), config=config, generator=FakeGenerator())


# ------------------------------------------------------------- validation
@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_text_rejected(pipeline, text):
    with pytest.raises(InputValidationError, match="empty"):
        pipeline.summarize(text)


def test_too_short_text_rejected(pipeline):
    with pytest.raises(InputValidationError, match="too short"):
        pipeline.summarize("hi there")


def test_very_long_text_rejected(config):
    pipe = PredictionPipeline(
        settings(max_input_chars=200), config=config, generator=FakeGenerator()
    )
    with pytest.raises(InputValidationError, match="too long") as info:
        pipe.summarize("word " * 100)
    assert info.value.context == {"chars": 499, "max_chars": 200}


def test_long_but_allowed_text_is_flagged_as_truncated(config):
    pipe = PredictionPipeline(settings(), config=config, generator=FakeGenerator(tokens=900))
    assert pipe.summarize("word " * 1000).truncated is True


# ---------------------------------------------------------------- results
def test_summary_metadata(pipeline):
    result = pipeline.summarize(f"  {DIALOGUE}  ")  # surrounding whitespace is ignored

    assert result.summary == "Anna and Bob meet at noon."
    assert result.original_length == len(DIALOGUE)
    assert result.summary_length == len(result.summary)
    assert result.compression_ratio == round(len(DIALOGUE) / len(result.summary), 2)
    assert result.original_word_count == 14 and result.summary_word_count == 6
    assert result.processing_time_seconds >= 0
    assert result.truncated is False
    assert result.model_version == "some/model"


def test_empty_model_output_does_not_divide_by_zero(config):
    pipe = PredictionPipeline(settings(), config=config, generator=FakeGenerator(summary=""))
    assert pipe.summarize(DIALOGUE).compression_ratio == len(DIALOGUE)


def test_batch_is_generated_in_one_call(pipeline):
    results, total = pipeline.summarize_batch([DIALOGUE, DIALOGUE + " Thanks!"])
    assert len(results) == 2 and total >= 0
    assert len(pipeline.generator.batches) == 1  # one batched generate() call


def test_batch_validation_reports_the_bad_index(pipeline):
    with pytest.raises(InputValidationError, match=r"texts\[1\]") as info:
        pipeline.summarize_batch([DIALOGUE, "  "])
    assert info.value.context["index"] == 1
    assert pipeline.generator.batches == []  # nothing generated: fail fast


def test_batch_size_limit(config):
    pipe = PredictionPipeline(settings(max_batch_size=2), config=config, generator=FakeGenerator())
    with pytest.raises(InputValidationError, match="Too many texts"):
        pipe.summarize_batch([DIALOGUE] * 3)
    with pytest.raises(InputValidationError, match="at least one"):
        pipe.summarize_batch([])


def test_model_not_loaded(config):
    pipe = PredictionPipeline(settings(), config=config, generator=FakeGenerator(loaded=False))
    assert not pipe.is_ready
    with pytest.raises(ModelNotReadyError):
        pipe.summarize(DIALOGUE)
    pipe.load()
    assert pipe.summarize(DIALOGUE).summary


def test_unknown_generation_profile(config):
    with pytest.raises(ConfigurationError, match="GENERATION_PROFILE"):
        PredictionPipeline(
            settings(generation_profile="nope"), config=config, generator=FakeGenerator()
        )


def test_info_exposes_limits(pipeline):
    info = pipeline.info()
    assert info["max_input_chars"] == 20_000 and info["min_input_words"] == 5
    assert info["generation_profile"] == "optimized" and info["model_loaded"] is True


# -------------------------------------------------------- model selection
def test_registry_without_production_falls_back(config):
    pipe = PredictionPipeline(
        settings(model_source="registry"), config=config, generator=FakeGenerator()
    )
    assert pipe.model_path == "some/model"


def test_registry_production_model_is_served(config):
    registry = ModelRegistry(config.registry_file)
    model_dir = config.registry_file.parent / "run-7"
    model_dir.mkdir(parents=True)
    registry.register("run-7", model_dir)
    registry.promote("run-7")

    pipe = PredictionPipeline(
        settings(model_source="registry"), config=config, generator=FakeGenerator()
    )
    assert pipe.model_path == str(model_dir)
    assert pipe.model_version == "run-7 (v1)"
    assert pipe.summarize(DIALOGUE).model_version == "run-7 (v1)"


def test_missing_production_files_fall_back(config):
    registry = ModelRegistry(config.registry_file)
    model_dir = config.registry_file.parent / "run-8"
    model_dir.mkdir(parents=True)
    registry.register("run-8", model_dir)
    registry.promote("run-8")
    model_dir.rmdir()  # e.g. a Docker image built without the weights

    pipe = PredictionPipeline(
        settings(model_source="registry"), config=config, generator=FakeGenerator()
    )
    assert pipe.model_path == "some/model"


@pytest.mark.slow
def test_end_to_end_with_tiny_model(config):
    pipe = PredictionPipeline(settings(model_name_or_path=TINY, device="cpu"), config=config).load()
    result = pipe.summarize(DIALOGUE + " <file_photo>")
    assert isinstance(result.summary, str) and result.input_tokens > 0
    results, _ = pipe.summarize_batch([DIALOGUE, DIALOGUE])
    assert len(results) == 2
