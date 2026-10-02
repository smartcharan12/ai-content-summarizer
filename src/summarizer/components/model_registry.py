"""A small file-based model registry (model versioning).

Every training run saves its model to ``artifacts/models/<run_id>/`` and is
recorded in ``artifacts/models/registry.json``::

    {
      "production": "20261002-153000",
      "models": [
        {"run_id": "20261002-153000", "version": 1, "stage": "production",
         "path": "artifacts/models/20261002-153000", "base_model": "google/pegasus-xsum",
         "metrics": {"rougeL": ...}, "git_commit": "...", "data_fingerprint": "...", ...}
      ]
    }

Stages: ``candidate`` (just trained) -> ``production`` (served by the API) ->
``archived`` (replaced). Rolling back means promoting an older run again.

Why not MLflow? For a single-developer project a JSON file you can read in a
code review is enough. The interface (register / promote / get_production) is
the same idea MLflow's Model Registry uses, so migrating later is easy.

CLI::

    python -m summarizer.components.model_registry list
    python -m summarizer.components.model_registry promote <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarizer.constants import PROJECT_ROOT
from summarizer.exception import ModelRegistryError, SummarizerError
from summarizer.logger import get_logger
from summarizer.utils.common import resolve_path

logger = get_logger(__name__)

STAGES = ("candidate", "production", "archived")


class ModelRegistry:
    """Read and update ``registry.json``.

    Args:
        registry_file: Location of the registry JSON (created on first write).
    """

    def __init__(self, registry_file: str | Path) -> None:
        self.registry_file = Path(registry_file)

    # ------------------------------------------------------------------ io
    def _load(self) -> dict[str, Any]:
        if not self.registry_file.is_file():
            return {"production": None, "models": []}
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ModelRegistryError(
                "Registry file is corrupt", context={"path": str(self.registry_file)}
            ) from err

    def _save(self, data: dict[str, Any]) -> None:
        """Atomic write: a crash mid-write can never leave a half-written registry."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.registry_file.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.registry_file)

    @staticmethod
    def _portable(path: Path) -> str:
        """Store paths relative to the project root, so the registry works in Docker too."""
        path = Path(path).resolve()
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------- queries
    def list_models(self) -> list[dict[str, Any]]:
        return self._load()["models"]

    def get(self, run_id: str) -> dict[str, Any]:
        for entry in self.list_models():
            if entry["run_id"] == run_id:
                return entry
        raise ModelRegistryError("Unknown run id", context={"run_id": run_id})

    def get_production(self) -> dict[str, Any] | None:
        """The entry currently marked production, or None if none is."""
        run_id = self._load()["production"]
        return self.get(run_id) if run_id else None

    def model_path(self, entry: dict[str, Any]) -> Path:
        return resolve_path(entry["path"])

    # ------------------------------------------------------------ updates
    def register(self, run_id: str, path: Path, **metadata: Any) -> dict[str, Any]:
        """Add a new run as a ``candidate``. Returns the stored entry."""
        data = self._load()
        if any(m["run_id"] == run_id for m in data["models"]):
            raise ModelRegistryError("Run id already registered", context={"run_id": run_id})
        entry = {
            "run_id": run_id,
            "version": len(data["models"]) + 1,
            "stage": "candidate",
            "path": self._portable(path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            **metadata,
        }
        data["models"].append(entry)
        self._save(data)
        logger.info("Registered model %s (version %d) as candidate", run_id, entry["version"])
        return entry

    def update_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        """Attach evaluation metrics to a run (used by the evaluation stage)."""
        data = self._load()
        for entry in data["models"]:
            if entry["run_id"] == run_id:
                entry["metrics"] = {**entry.get("metrics", {}), **metrics}
                self._save(data)
                return
        raise ModelRegistryError("Unknown run id", context={"run_id": run_id})

    def promote(self, run_id: str, allow_smoke: bool = False) -> dict[str, Any]:
        """Make ``run_id`` the production model; the previous one is archived."""
        data = self._load()
        target = next((m for m in data["models"] if m["run_id"] == run_id), None)
        if target is None:
            raise ModelRegistryError("Unknown run id", context={"run_id": run_id})
        if target.get("smoke_test") and not allow_smoke:
            raise ModelRegistryError(
                "Refusing to promote a smoke-test model (random weights)",
                context={"run_id": run_id},
            )
        if not resolve_path(target["path"]).is_dir():
            raise ModelRegistryError("Model files are missing", context={"path": target["path"]})

        for entry in data["models"]:
            if entry["stage"] == "production" and entry["run_id"] != run_id:
                entry["stage"] = "archived"
        target["stage"] = "production"
        target["promoted_at"] = datetime.now(timezone.utc).isoformat()
        data["production"] = run_id
        self._save(data)
        logger.info("Promoted %s to production", run_id)
        return target


def main(argv: list[str] | None = None) -> int:
    from summarizer.config.configuration import ConfigurationManager

    parser = argparse.ArgumentParser(description="Inspect or update the model registry.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show all registered models")
    promote = sub.add_parser("promote", help="Mark a run as the production model")
    promote.add_argument("run_id")
    args = parser.parse_args(argv)

    registry = ModelRegistry(ConfigurationManager().registry_file)
    try:
        if args.command == "list":
            models = registry.list_models()
            if not models:
                print("No models registered yet.")
            for m in models:
                rouge = m.get("metrics", {}).get("rougeL")
                rouge_str = (
                    f"rougeL={rouge:.4f}" if isinstance(rouge, int | float) else "rougeL=n/a"
                )
                print(
                    f"v{m['version']:<3} {m['run_id']}  {m['stage']:<10} {rouge_str}  {m['path']}"
                )
        else:
            registry.promote(args.run_id)
    except SummarizerError as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
