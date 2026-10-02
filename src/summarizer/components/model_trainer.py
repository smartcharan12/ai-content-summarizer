"""Stage 4 - Model training (fine-tuning PEGASUS on SAMSum).

How fine-tuning works here, briefly:

* Start from pretrained weights (``google/pegasus-xsum``), which already
  "know" English and how to summarise news.
* Show the model (dialogue, summary) pairs. For each one it predicts the
  summary token by token; the loss is the cross-entropy between its
  predictions and the real summary tokens (padding positions = -100 are
  ignored). Back-propagation nudges every weight to reduce that loss.
* After each epoch, compute the loss on the validation split. The best
  checkpoint (lowest ``eval_loss``) is kept, and early stopping ends training
  once validation loss stops improving (a sign of over-fitting).

Hugging Face ``Seq2SeqTrainer`` runs the loop: batching, gradient
accumulation, mixed precision, checkpointing, LR schedule. This class turns
our typed config into its arguments, guards against common mistakes, and
records everything needed to reproduce the run.

ROUGE is deliberately *not* computed during training: generating summaries
for every eval step is slow, and eval_loss is enough to pick a checkpoint.
Stage 5 (model evaluation) computes ROUGE once on the final model.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarizer.components.data_transformation import META_FILE
from summarizer.components.model_registry import ModelRegistry
from summarizer.constants import PROJECT_ROOT
from summarizer.entity.config_entity import ModelTrainerConfig
from summarizer.exception import ConfigurationError, ModelTrainingError
from summarizer.logger import get_logger
from summarizer.utils.common import load_json, save_json, set_seed

logger = get_logger(__name__)

# TrainingParams fields that are ours, not Seq2SeqTrainingArguments arguments.
_NON_HF_FIELDS = {
    "early_stopping_patience",
    "mixed_precision",
    "max_train_samples",
    "max_eval_samples",
    "keep_checkpoints",
}


def git_commit() -> str | None:
    """Current git commit (with a '-dirty' suffix if there are uncommitted changes)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_bf16(mode: str) -> bool:
    """Decide whether to train in bfloat16.

    bf16 has the same exponent range as fp32, so unlike fp16 it does not
    overflow on PEGASUS. Only GPUs with compute capability >= 8.0 (Ampere:
    A100, L4, RTX 30xx and newer) support it natively; a T4 (7.5) trains in fp32.
    """
    import torch

    if mode == "no":
        return False
    native = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if mode == "bf16" and not native:
        raise ConfigurationError(
            "mixed_precision=bf16 requested but this device has no native bf16 support",
            context={"cuda": torch.cuda.is_available()},
        )
    return native


class ModelTrainer:
    """Fine-tune a seq2seq model on the processed dataset.

    Args:
        config: Stage configuration (paths, base model, hyperparameters).
        registry: Where to record the trained model (defaults to the configured registry).
    """

    def __init__(self, config: ModelTrainerConfig, registry: ModelRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ModelRegistry(config.registry_file)

    # ------------------------------------------------------------ checks
    def check_processed_data(self) -> dict[str, Any]:
        """Make sure the tokenized data matches the model we're about to train."""
        meta_path = self.config.processed_dir / META_FILE
        if not meta_path.is_file():
            raise ModelTrainingError(
                "Processed dataset not found - run the transformation stage first",
                context={"expected": str(meta_path)},
            )
        meta = load_json(meta_path)
        mismatches = {
            key: (meta.get(key), expected)
            for key, expected in (
                ("tokenizer_name", self.config.base_model),
                ("max_input_length", self.config.max_input_length),
                ("max_target_length", self.config.max_target_length),
            )
            if meta.get(key) != expected
        }
        if mismatches:
            raise ModelTrainingError(
                "Processed data was built with different settings - re-run "
                "`--stages transformation` (found, expected)",
                context=mismatches,
            )
        return meta

    def data_fingerprint(self) -> str:
        """Fingerprint of the processed dataset (changes whenever the data does)."""
        import hashlib

        digest = hashlib.sha256((self.config.processed_dir / META_FILE).read_bytes())
        for state in sorted(self.config.processed_dir.glob("*/state.json")):
            digest.update(json.loads(state.read_text()).get("_fingerprint", "").encode())
        return digest.hexdigest()[:16]

    # ------------------------------------------------------- components
    def load_datasets(self) -> tuple[Any, Any]:
        from datasets import load_from_disk

        data = load_from_disk(str(self.config.processed_dir))
        train, evaluation = data["train"], data["validation"]
        tp = self.config.training
        if tp.max_train_samples:
            train = train.shuffle(seed=self.config.seed).select(
                range(min(tp.max_train_samples, len(train)))
            )
        if tp.max_eval_samples:
            evaluation = evaluation.shuffle(seed=self.config.seed).select(
                range(min(tp.max_eval_samples, len(evaluation)))
            )
        logger.info("Training rows: %d | validation rows: %d", len(train), len(evaluation))
        return train, evaluation

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading base model '%s'", self.config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model)

        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions and self.config.max_input_length > max_positions:
            raise ConfigurationError(
                "max_input_length exceeds what the model supports",
                context={
                    "max_input_length": self.config.max_input_length,
                    "model_max": max_positions,
                },
            )
        if self.config.training.gradient_checkpointing:
            model.config.use_cache = False  # the KV cache is incompatible with checkpointing
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model has %.1fM parameters", n_params / 1e6)
        return model, tokenizer

    def build_training_args(self, output_dir: Path) -> Any:
        import torch
        from transformers import Seq2SeqTrainingArguments

        hf_kwargs = {
            k: v for k, v in asdict(self.config.training).items() if k not in _NON_HF_FIELDS
        }
        return Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            bf16=resolve_bf16(self.config.training.mixed_precision),
            seed=self.config.seed,
            report_to="none",  # no W&B/TensorBoard account needed; logs go to our files
            predict_with_generate=False,  # eval_loss only; ROUGE is computed in Stage 5
            dataloader_pin_memory=torch.cuda.is_available(),
            **hf_kwargs,
        )

    # --------------------------------------------------------------- run
    def train(self, run_id: str | None = None, resume: bool = False) -> dict[str, Any]:
        """Train, save, and register a model.

        Args:
            run_id: Identifier for this run (default: UTC timestamp). Pass the id
                of an interrupted run together with ``resume=True`` to continue it.
            resume: Continue from the newest checkpoint of ``run_id`` (useful when
                a free Colab/Kaggle session disconnects).

        Returns:
            The registry entry of the new model.
        """
        from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer
        from transformers.trainer_utils import get_last_checkpoint

        if resume and not run_id:
            raise ModelTrainingError("resume=True needs the run_id of the interrupted run")
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = self.config.checkpoints_dir / run_id
        final_dir = self.config.models_dir / run_id
        if final_dir.exists():
            raise ModelTrainingError(
                "A model with this run id already exists", context={"run_id": run_id}
            )

        resume_from = None
        if resume:
            resume_from = (
                get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.is_dir() else None
            )
            if resume_from is None:
                raise ModelTrainingError(
                    "No checkpoint to resume from", context={"dir": str(checkpoint_dir)}
                )
            logger.info("Resuming from %s", resume_from)

        self.check_processed_data()
        set_seed(self.config.seed)
        train_ds, eval_ds = self.load_datasets()
        model, tokenizer = self.load_model_and_tokenizer()
        args = self.build_training_args(checkpoint_dir)
        tp = self.config.training

        callbacks = []
        if tp.early_stopping_patience and tp.load_best_model_at_end:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=tp.early_stopping_patience)
            )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            # Pads each batch to its longest row; labels are padded with -100.
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, model=model, pad_to_multiple_of=8 if args.bf16 else None
            ),
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        logger.info(
            "Training: epochs=%s lr=%s effective_batch=%d bf16=%s optim=%s device=%s",
            tp.num_train_epochs,
            tp.learning_rate,
            tp.effective_batch_size,
            args.bf16,
            tp.optim,
            args.device,
        )
        start = time.perf_counter()
        try:
            train_result = trainer.train(resume_from_checkpoint=resume_from)
            eval_metrics = trainer.evaluate()
        except Exception as err:
            raise ModelTrainingError(
                "Training failed", context={"run_id": run_id, "error": str(err)}
            ) from err
        duration = time.perf_counter() - start

        # Saves weights (safetensors), config, generation config and tokenizer.
        trainer.save_model(str(final_dir))
        history = trainer.state.log_history
        metadata = self._metadata(
            run_id, train_result.metrics, eval_metrics, duration, args, trainer
        )
        save_json(final_dir / "metadata.json", metadata)
        save_json(final_dir / "log_history.json", history)
        save_json(
            self.config.reports_dir / f"training_{run_id}.json",
            {**metadata, "log_history": history},
        )

        if not tp.keep_checkpoints and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info("Deleted intermediate checkpoints in %s", checkpoint_dir)

        entry = self.registry.register(
            run_id,
            final_dir,
            base_model=self.config.base_model,
            smoke_test=self.config.smoke_test,
            git_commit=metadata["git_commit"],
            data_fingerprint=metadata["data_fingerprint"],
            metrics={"eval_loss": metadata["eval_loss"]},
        )
        logger.info("Model saved to %s (eval_loss=%.4f)", final_dir, metadata["eval_loss"])
        return entry

    def _metadata(
        self, run_id, train_metrics, eval_metrics, duration, args, trainer
    ) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "run_id": run_id,
            "base_model": self.config.base_model,
            "smoke_test": self.config.smoke_test,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "data_fingerprint": self.data_fingerprint(),
            "train_rows": len(trainer.train_dataset),
            "eval_rows": len(trainer.eval_dataset),
            "hyperparameters": asdict(self.config.training),
            "max_input_length": self.config.max_input_length,
            "max_target_length": self.config.max_target_length,
            "bf16": args.bf16,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "epochs_completed": trainer.state.epoch,
            "global_steps": trainer.state.global_step,
            "train_loss": train_metrics.get("train_loss"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "training_seconds": round(duration, 1),
            "environment": {
                "device": str(args.device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
