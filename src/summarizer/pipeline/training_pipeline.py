"""Training pipeline: runs the stages in order.

Usage (from the project root, venv active)::

    python -m summarizer.pipeline.training_pipeline                  # all available stages
    python -m summarizer.pipeline.training_pipeline --stages ingestion validation
    python -m summarizer.pipeline.training_pipeline --force-download
    python -m summarizer.pipeline.training_pipeline --stages training --run-id <id> --resume
    SMOKE_TEST=1 python -m summarizer.pipeline.training_pipeline     # tiny/fast run

Each stage reads the previous stage's files from disk, so you can re-run a
single stage (e.g. ``--stages validation`` after changing a rule) without
repeating the earlier ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

from summarizer.components.data_ingestion import DataIngestion
from summarizer.components.data_transformation import DataTransformation
from summarizer.components.data_validation import DataValidation
from summarizer.components.model_evaluation import ModelEvaluator
from summarizer.components.model_trainer import ModelTrainer
from summarizer.config.configuration import ConfigurationManager
from summarizer.exception import SummarizerError
from summarizer.logger import get_logger
from summarizer.utils.common import set_seed

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the five training-pipeline stages."""

    def __init__(
        self,
        config: ConfigurationManager | None = None,
        force_download: bool = False,
        run_id: str | None = None,
        resume: bool = False,
    ):
        self.config = config or ConfigurationManager()
        self.force_download = force_download
        self.run_id = run_id
        self.resume = resume
        self.stages: dict[str, Callable[[], object]] = {
            "ingestion": self.run_ingestion,
            "validation": self.run_validation,
            "transformation": self.run_transformation,
            "training": self.run_training,
            "evaluation": self.run_evaluation,
        }

    def run_ingestion(self) -> object:
        return DataIngestion(self.config.get_data_ingestion_config()).run(force=self.force_download)

    def run_validation(self) -> object:
        return DataValidation(self.config.get_data_validation_config()).run()

    def run_transformation(self) -> object:
        return DataTransformation(self.config.get_data_transformation_config()).run()

    def run_training(self) -> object:
        trainer = ModelTrainer(self.config.get_model_trainer_config())
        entry = trainer.train(run_id=self.run_id, resume=self.resume)
        self.run_id = entry["run_id"]  # so the evaluation stage scores this exact model
        return entry

    def run_evaluation(self) -> object:
        # run_id None = most recently registered model
        return ModelEvaluator(
            self.config.get_model_evaluation_config(), model_ref=self.run_id
        ).run()

    def run(self, stages: list[str] | None = None) -> None:
        """Run the selected stages (default: all) in pipeline order."""
        selected = stages or list(self.stages)
        unknown = [s for s in selected if s not in self.stages]
        if unknown:
            raise ValueError(f"Unknown stage(s) {unknown}; choose from {list(self.stages)}")

        set_seed(self.config.seed)
        for name in self.stages:  # iterate in pipeline order, not CLI order
            if name not in selected:
                continue
            logger.info("========== Stage: %s ==========", name)
            start = time.perf_counter()
            self.stages[name]()
            logger.info("Stage '%s' finished in %.1fs", name, time.perf_counter() - start)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the summarization training pipeline.")
    parser.add_argument("--stages", nargs="+", help="Stages to run (default: all).")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download raw data even if present."
    )
    parser.add_argument("--run-id", help="Name for the training run (default: UTC timestamp).")
    parser.add_argument(
        "--resume", action="store_true", help="Resume the training run given by --run-id."
    )
    args = parser.parse_args(argv)

    try:
        TrainingPipeline(
            force_download=args.force_download, run_id=args.run_id, resume=args.resume
        ).run(args.stages)
    except (SummarizerError, ValueError) as err:
        logger.error("Pipeline failed: %s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
