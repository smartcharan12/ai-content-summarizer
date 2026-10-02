"""Stage 5 - Model evaluation and decoding experiments.

``ModelEvaluator.run()`` scores one model with one decoding profile on a fixed,
seeded sample of the *test* split and reports:

* **Quality:** ROUGE-1/2/L/Lsum (F1 x 100) with 95% bootstrap confidence
  intervals, plus BLEU.
* **Behaviour:** average summary length, compression ratio (dialogue words /
  summary words), share of summaries that repeat a trigram, empty outputs.
* **Speed:** single-request latency (batch size 1, what an API user waits
  for), p50 and p95, plus batched throughput.

Results go to ``evaluation_metrics.json`` (latest run), to
``evaluations/<model>__<profile>.json`` (history: baseline and fine-tuned side
by side), and to ``evaluation_samples.csv`` (every row: dialogue, generated
summary, reference, per-row ROUGE). If the model is in the registry, its
metrics are stored there. With ``auto_promote`` on, it becomes production when
it beats the current production model on the *same* test rows.

``ModelEvaluator.run_experiments()`` runs the decoding grid from
``params.yaml``: the same model on the same rows, changing one generation
parameter at a time.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from summarizer.components.model_registry import ModelRegistry
from summarizer.components.summary_generator import SummaryGenerator
from summarizer.components.text_cleaning import TextCleaner
from summarizer.entity.config_entity import GenerationParams, ModelEvaluationConfig
from summarizer.exception import ModelEvaluationError
from summarizer.logger import get_logger
from summarizer.utils.common import resolve_path, save_json
from summarizer.utils.metrics import (
    ROUGE_TYPES,
    bleu_score,
    bootstrap_ci,
    compression_ratio,
    has_repeated_ngram,
    rouge_scores,
    word_count,
)

logger = get_logger(__name__)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


class ModelEvaluator:
    """Evaluate a model from the registry, a local folder or the Hugging Face Hub.

    Args:
        config: Evaluation configuration.
        model_ref: A registry run id, a model folder, or a Hub id such as
            ``google/pegasus-xsum`` (zero-shot baseline). ``None`` = the most
            recently registered model.
        generator: Optional pre-built :class:`SummaryGenerator` (for tests).
        device: ``auto``/``cpu``/``cuda``/``mps``. Use ``cpu`` on a GPU machine to
            measure the latency a CPU server (e.g. EC2 without GPU) would have.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_ref: str | None = None,
        generator: SummaryGenerator | None = None,
        registry: ModelRegistry | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.device = device
        self.registry = registry or ModelRegistry(config.registry_file)
        self.model_path, self.model_label, self.registry_entry = self._resolve_model(model_ref)
        self._generator = generator

    # ------------------------------------------------------------ setup
    def _resolve_model(self, ref: str | None) -> tuple[str, str, dict[str, Any] | None]:
        models = self.registry.list_models()
        if ref is None:
            if not models:
                raise ModelEvaluationError(
                    "No trained model is registered. Train one, or evaluate a Hub model "
                    "directly, e.g. --model google/pegasus-xsum"
                )
            entry = models[-1]
            return str(self.registry.model_path(entry)), entry["run_id"], entry
        for entry in models:
            if entry["run_id"] == ref:
                return str(self.registry.model_path(entry)), ref, entry
        local = resolve_path(ref)
        return (str(local) if local.is_dir() else ref), ref, None

    @property
    def generator(self) -> SummaryGenerator:
        if self._generator is None:
            cleaner = TextCleaner(
                normalize_whitespace=self.config.normalize_whitespace,
                normalize_tags=self.config.normalize_tags,
                lowercase=self.config.lowercase,
            )
            self._generator = SummaryGenerator(
                self.model_path,
                device=self.device,
                max_input_length=self.config.max_input_length,
                cleaner=cleaner,
            ).load()
        return self._generator

    def load_eval_set(self, num_samples: int) -> tuple[pd.DataFrame, str]:
        """A seeded sample of the test split, plus a signature of its row ids."""
        path = self.config.validated_dir / "test.csv"
        if not path.is_file():
            raise ModelEvaluationError(
                "Validated test split not found - run ingestion and validation first",
                context={"path": str(path)},
            )
        df = pd.read_csv(path, dtype=object, keep_default_na=False)
        if num_samples < len(df):
            df = df.sample(n=num_samples, random_state=self.config.seed)
        df = df.reset_index(drop=True)
        signature = hashlib.sha256(",".join(df["id"]).encode()).hexdigest()[:12]
        return df, signature

    # ---------------------------------------------------------- scoring
    def score(
        self, df: pd.DataFrame, params: GenerationParams
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Generate summaries for ``df`` and compute every metric."""
        dialogues, references = df["dialogue"].tolist(), df["summary"].tolist()
        start = time.perf_counter()
        predictions = self.generator.generate(dialogues, params, batch_size=self.config.batch_size)
        elapsed = time.perf_counter() - start

        per_row = rouge_scores(predictions, references)
        rows = pd.DataFrame(
            {
                "id": df["id"],
                "dialogue": dialogues,
                "generated_summary": predictions,
                "reference_summary": references,
                **{t: np.round(v, 4) for t, v in per_row.items()},
                "generated_words": [word_count(p) for p in predictions],
                "reference_words": [word_count(r) for r in references],
                "compression_ratio": [
                    round(compression_ratio(d, p), 2)
                    for d, p in zip(dialogues, predictions, strict=True)
                ],
            }
        )
        metrics: dict[str, Any] = {
            t: round(100 * float(np.mean(per_row[t])), 2) for t in ROUGE_TYPES
        }
        metrics["confidence_intervals_95"] = {
            t: [
                round(100 * x, 2)
                for x in bootstrap_ci(per_row[t], self.config.bootstrap_resamples, self.config.seed)
            ]
            for t in ROUGE_TYPES
        }
        metrics["bleu"] = round(bleu_score(predictions, references), 2)
        metrics["avg_generated_words"] = round(float(rows["generated_words"].mean()), 2)
        metrics["avg_reference_words"] = round(float(rows["reference_words"].mean()), 2)
        metrics["median_compression_ratio"] = round(float(rows["compression_ratio"].median()), 2)
        metrics["reference_median_compression_ratio"] = round(
            float(
                np.median(
                    [compression_ratio(d, r) for d, r in zip(dialogues, references, strict=True)]
                )
            ),
            2,
        )
        metrics["repeated_trigram_rate"] = round(
            float(np.mean([has_repeated_ngram(p, 3) for p in predictions])), 4
        )
        metrics["empty_outputs"] = int(sum(1 for p in predictions if not p.strip()))
        metrics["batched_seconds_per_sample"] = round(elapsed / max(len(df), 1), 4)
        return metrics, rows

    def measure_latency(self, df: pd.DataFrame, params: GenerationParams) -> dict[str, Any]:
        """Single-request latency: one dialogue at a time, like an API call."""
        n = min(self.config.latency_samples, len(df))
        if n == 0:
            return {}
        texts = df["dialogue"].tolist()[:n]
        self.generator.generate(texts[:1], params, batch_size=1)  # warm-up (first call is slower)
        timings = []
        for text in texts:
            start = time.perf_counter()
            self.generator.generate([text], params, batch_size=1)
            timings.append(time.perf_counter() - start)
        return {
            "requests": n,
            "p50_seconds": round(float(np.percentile(timings, 50)), 3),
            "p95_seconds": round(float(np.percentile(timings, 95)), 3),
            "mean_seconds": round(float(np.mean(timings)), 3),
            "device": self.generator.device,
        }

    # -------------------------------------------------------------- run
    def run(
        self,
        profile: str | None = None,
        num_samples: int | None = None,
        promote: bool | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        profile = profile or cfg.default_profile
        if profile not in cfg.generation_profiles:
            raise ModelEvaluationError(
                "Unknown generation profile",
                context={"profile": profile, "available": sorted(cfg.generation_profiles)},
            )
        params = cfg.generation_profiles[profile]
        df, signature = self.load_eval_set(num_samples or cfg.num_samples)
        logger.info(
            "Evaluating '%s' (profile=%s) on %d test rows", self.model_label, profile, len(df)
        )

        metrics, rows = self.score(df, params)
        latency = self.measure_latency(df, params)
        result = {
            "model": self.model_label,
            "model_path": self.model_path,
            "registry_run_id": self.registry_entry["run_id"] if self.registry_entry else None,
            "smoke_test": cfg.smoke_test,
            "profile": profile,
            "generation_params": params.to_dict(),
            "eval_set": {
                "split": "test",
                "rows": len(df),
                "seed": cfg.seed,
                "signature": signature,
            },
            "metrics": metrics,
            "latency": latency,
            "model_load_seconds": self.generator.load_seconds,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "examples": rows.head(cfg.num_examples_to_save)[
                ["dialogue", "generated_summary", "reference_summary", "rougeL"]
            ].to_dict(orient="records"),
        }

        save_json(cfg.metrics_file, result)
        history_name = f"{slugify(self.model_label)}__{profile}__{self.generator.device}"
        if num_samples:  # non-standard row count: keep it apart from the canonical result
            history_name += f"__n{len(df)}"
        save_json(cfg.history_dir / f"{history_name}.json", result)
        cfg.samples_file.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(cfg.samples_file, index=False)
        logger.info(
            "ROUGE-1 %.2f | ROUGE-2 %.2f | ROUGE-L %.2f | ROUGE-Lsum %.2f | BLEU %.2f | p50 %ss",
            metrics["rouge1"],
            metrics["rouge2"],
            metrics["rougeL"],
            metrics["rougeLsum"],
            metrics["bleu"],
            latency.get("p50_seconds"),
        )

        # Only the canonical evaluation (default row count) is recorded in the
        # registry, so promotion always compares models on identical rows.
        if self.registry_entry and not num_samples:
            self.registry.update_metrics(
                self.registry_entry["run_id"],
                {
                    **{t: metrics[t] for t in ROUGE_TYPES},
                    "bleu": metrics["bleu"],
                    "profile": profile,
                    "eval_signature": signature,
                    "eval_rows": len(df),
                },
            )
            result["promotion"] = self._maybe_promote(
                metrics["rougeL"],
                signature,
                profile,
                cfg.auto_promote if promote is None else promote,
            )
        return result

    def _maybe_promote(self, rouge_l: float, signature: str, profile: str, enabled: bool) -> str:
        entry = self.registry_entry
        if not enabled:
            decision = "skipped: auto_promote disabled"
        elif entry.get("smoke_test"):
            decision = "skipped: smoke-test model"
        else:
            production = self.registry.get_production()
            if production is None:
                self.registry.promote(entry["run_id"])
                decision = "promoted: no production model existed"
            elif production["run_id"] == entry["run_id"]:
                decision = "unchanged: already production"
            else:
                prod = production.get("metrics", {})
                if prod.get("eval_signature") != signature or prod.get("profile") != profile:
                    decision = (
                        "skipped: production was evaluated on different rows/profile - "
                        "re-evaluate it with the same settings to compare"
                    )
                elif rouge_l > prod.get("rougeL", float("-inf")):
                    self.registry.promote(entry["run_id"])
                    decision = f"promoted: ROUGE-L {rouge_l} > {prod['rougeL']}"
                else:
                    decision = f"kept current production: ROUGE-L {rouge_l} <= {prod['rougeL']}"
        logger.info("Promotion decision for %s: %s", entry["run_id"], decision)
        return decision

    # ------------------------------------------------------ experiments
    def run_experiments(self, num_samples: int | None = None) -> pd.DataFrame:
        """Score every decoding configuration on the same rows; save a results table."""
        cfg = self.config
        if not cfg.experiments:
            raise ModelEvaluationError("No experiments configured in params.yaml")
        df, signature = self.load_eval_set(num_samples or cfg.experiment_samples)
        logger.info("Running %d decoding experiments on %d rows", len(cfg.experiments), len(df))

        records = []
        for run in cfg.experiments:
            metrics, _ = self.score(df, run.params)
            p = run.params
            records.append(
                {
                    "experiment": run.name,
                    "num_beams": p.num_beams,
                    "length_penalty": p.length_penalty if p.num_beams > 1 else None,
                    "no_repeat_ngram_size": p.no_repeat_ngram_size,
                    "min_length": p.min_length,
                    "max_length": p.max_length,
                    "early_stopping": p.early_stopping if p.num_beams > 1 else None,
                    **{t: metrics[t] for t in ROUGE_TYPES},
                    "rougeL_ci95": metrics["confidence_intervals_95"]["rougeL"],
                    "bleu": metrics["bleu"],
                    "avg_words": metrics["avg_generated_words"],
                    "repeated_trigram_rate": metrics["repeated_trigram_rate"],
                    "sec_per_sample": metrics["batched_seconds_per_sample"],
                }
            )
            logger.info(
                "%-24s ROUGE-L %.2f  words %.1f",
                run.name,
                metrics["rougeL"],
                metrics["avg_generated_words"],
            )

        table = pd.DataFrame(records)
        cfg.experiments_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cfg.experiments_file, index=False)
        header = (
            f"<!-- model: {self.model_label} | test rows: {len(df)} (signature {signature}, "
            f"seed {cfg.seed}) | device: {self.generator.device} | "
            f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC -->\n\n"
        )
        cfg.experiments_file.with_suffix(".md").write_text(
            header
            + table.astype(object)
            .where(table.notna(), "—")
            .to_markdown(index=False, floatfmt=".2f"),
            encoding="utf-8",
        )
        logger.info("Saved experiment table to %s", cfg.experiments_file)
        return table
