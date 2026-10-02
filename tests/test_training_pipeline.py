"""Tests for the TrainingPipeline orchestrator (stages stubbed out)."""

from __future__ import annotations

import pytest

from summarizer.exception import DataValidationError
from summarizer.pipeline import training_pipeline as tp


@pytest.fixture
def pipeline(real_config_paths):
    from summarizer.config import ConfigurationManager

    p = tp.TrainingPipeline(ConfigurationManager(*real_config_paths, smoke_test=True))
    calls: list[str] = []
    p.stages = {name: (lambda n=name: calls.append(n)) for name in p.stages}
    p.calls = calls
    return p


def test_runs_all_stages_in_order(pipeline):
    pipeline.run()
    assert pipeline.calls == [
        "ingestion",
        "validation",
        "transformation",
        "training",
        "evaluation",
    ]


def test_stage_order_ignores_cli_order(pipeline):
    pipeline.run(["training", "ingestion"])
    assert pipeline.calls == ["ingestion", "training"]


def test_unknown_stage_rejected(pipeline):
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.run(["training-typo"])


def test_main_returns_nonzero_on_pipeline_error(monkeypatch):
    def failing_run(self, stages=None):
        raise DataValidationError("boom")

    monkeypatch.setattr(tp.TrainingPipeline, "run", failing_run)
    assert tp.main(["--stages", "validation"]) == 1
