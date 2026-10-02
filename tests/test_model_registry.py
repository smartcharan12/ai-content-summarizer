"""Tests for the file-based ModelRegistry."""

from __future__ import annotations

import pytest

from summarizer.components import model_registry as mr
from summarizer.components.model_registry import ModelRegistry
from summarizer.exception import ModelRegistryError


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models" / "registry.json")


def make_model_dir(tmp_path, name):
    path = tmp_path / "models" / name
    path.mkdir(parents=True)
    return path


def test_empty_registry(registry):
    assert registry.list_models() == []
    assert registry.get_production() is None


def test_register_assigns_versions_as_candidates(registry, tmp_path):
    first = registry.register("run-a", make_model_dir(tmp_path, "run-a"), base_model="m")
    second = registry.register("run-b", make_model_dir(tmp_path, "run-b"), base_model="m")

    assert (first["version"], second["version"]) == (1, 2)
    assert {m["stage"] for m in registry.list_models()} == {"candidate"}
    assert registry.get("run-b")["base_model"] == "m"


def test_duplicate_run_id_rejected(registry, tmp_path):
    path = make_model_dir(tmp_path, "run-a")
    registry.register("run-a", path)
    with pytest.raises(ModelRegistryError, match="already registered"):
        registry.register("run-a", path)


def test_promote_archives_previous_production(registry, tmp_path):
    registry.register("run-a", make_model_dir(tmp_path, "run-a"))
    registry.register("run-b", make_model_dir(tmp_path, "run-b"))

    registry.promote("run-a")
    assert registry.get_production()["run_id"] == "run-a"

    registry.promote("run-b")
    assert registry.get_production()["run_id"] == "run-b"
    assert registry.get("run-a")["stage"] == "archived"

    registry.promote("run-a")  # rollback
    assert registry.get("run-a")["stage"] == "production"
    assert registry.get("run-b")["stage"] == "archived"


def test_smoke_models_cannot_be_promoted_by_default(registry, tmp_path):
    registry.register("smoke", make_model_dir(tmp_path, "smoke"), smoke_test=True)
    with pytest.raises(ModelRegistryError, match="smoke-test"):
        registry.promote("smoke")
    registry.promote("smoke", allow_smoke=True)
    assert registry.get_production()["run_id"] == "smoke"


def test_promote_requires_model_files(registry, tmp_path):
    registry.register("gone", tmp_path / "models" / "gone")  # directory never created
    with pytest.raises(ModelRegistryError, match="missing"):
        registry.promote("gone")


def test_unknown_run_id(registry):
    with pytest.raises(ModelRegistryError, match="Unknown run id"):
        registry.promote("nope")
    with pytest.raises(ModelRegistryError):
        registry.update_metrics("nope", {"rougeL": 0.1})


def test_update_metrics_merges(registry, tmp_path):
    registry.register("run-a", make_model_dir(tmp_path, "run-a"), metrics={"eval_loss": 1.5})
    registry.update_metrics("run-a", {"rougeL": 0.4})
    assert registry.get("run-a")["metrics"] == {"eval_loss": 1.5, "rougeL": 0.4}


def test_paths_inside_project_are_stored_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(mr, "PROJECT_ROOT", tmp_path)
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    entry = registry.register("run-a", make_model_dir(tmp_path, "run-a"))
    assert entry["path"] == "models/run-a"


def test_corrupt_registry_file(registry):
    registry.registry_file.parent.mkdir(parents=True)
    registry.registry_file.write_text("{not json")
    with pytest.raises(ModelRegistryError, match="corrupt"):
        registry.list_models()


def test_no_temp_files_left_behind(registry, tmp_path):
    registry.register("run-a", make_model_dir(tmp_path, "run-a"))
    assert not list(registry.registry_file.parent.glob("*.tmp"))
