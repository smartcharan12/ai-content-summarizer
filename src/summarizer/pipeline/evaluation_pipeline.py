"""Evaluate models and run decoding experiments from the command line.

Examples (project root, venv active)::

    # Evaluate the most recently trained model with the default profile
    python -m summarizer.pipeline.evaluation_pipeline

    # Zero-shot baseline: the pretrained checkpoint, before any fine-tuning
    python -m summarizer.pipeline.evaluation_pipeline --model google/pegasus-xsum

    # A specific registered run, a different profile, fewer rows (-n runs are not
    # written to the registry, so they never affect promotion)
    python -m summarizer.pipeline.evaluation_pipeline --model <run_id> --profile baseline -n 50

    # CPU latency on a GPU machine (what a CPU-only EC2 server would see)
    python -m summarizer.pipeline.evaluation_pipeline --model <run_id> --device cpu -n 20

    # Decoding-parameter experiments (results table in artifacts/reports/)
    python -m summarizer.pipeline.evaluation_pipeline --model <run_id> --experiments
"""

from __future__ import annotations

import argparse
import sys

from summarizer.components.model_evaluation import ModelEvaluator
from summarizer.config.configuration import ConfigurationManager
from summarizer.exception import SummarizerError
from summarizer.logger import get_logger
from summarizer.utils.common import set_seed

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a summarization model.")
    parser.add_argument(
        "--model", help="Registry run id, model folder, or Hub id (default: latest run)."
    )
    parser.add_argument(
        "--profile", help="Generation profile from params.yaml (default: default_profile)."
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, help="Number of test rows (default from params)."
    )
    parser.add_argument(
        "--experiments", action="store_true", help="Run the decoding experiment grid."
    )
    parser.add_argument(
        "--no-promote", action="store_true", help="Never change the production model."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Run on this device (cpu = latency a CPU-only server would see).",
    )
    args = parser.parse_args(argv)

    manager = ConfigurationManager()
    set_seed(manager.seed)
    try:
        evaluator = ModelEvaluator(
            manager.get_model_evaluation_config(), model_ref=args.model, device=args.device
        )
        if args.experiments:
            table = evaluator.run_experiments(num_samples=args.num_samples)
            print(table.to_string(index=False))
        else:
            result = evaluator.run(
                profile=args.profile,
                num_samples=args.num_samples,
                promote=False if args.no_promote else None,
            )
            if "promotion" in result:
                print("Promotion:", result["promotion"])
    except SummarizerError as err:
        logger.error("Evaluation failed: %s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
