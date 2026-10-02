"""Tests for shared utilities, logging and the exception hierarchy."""

from __future__ import annotations

import json
import logging

import pytest

from summarizer.constants import PROJECT_ROOT
from summarizer.exception import (
    ConfigurationError,
    InputValidationError,
    SummarizerError,
)
from summarizer.logger import JsonFormatter, get_logger
from summarizer.utils import (
    create_directories,
    deep_merge,
    get_device,
    load_json,
    read_yaml,
    resolve_path,
    save_json,
    set_seed,
)


def test_resolve_path_relative_and_absolute(tmp_path):
    assert resolve_path("data/raw") == PROJECT_ROOT / "data" / "raw"
    assert resolve_path(tmp_path) == tmp_path


def test_read_yaml_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="empty"):
        read_yaml(empty)


def test_read_yaml_rejects_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        read_yaml(bad)


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3}, "c": 4}
    merged = deep_merge(base, override)

    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_json_roundtrip_creates_parents(tmp_path):
    target = tmp_path / "nested" / "dir" / "out.json"
    save_json(target, {"rouge1": 0.5, "text": "héllo"})
    assert load_json(target) == {"rouge1": 0.5, "text": "héllo"}


def test_create_directories(tmp_path):
    dirs = [tmp_path / "a" / "b", tmp_path / "c"]
    create_directories(dirs)
    assert all(d.is_dir() for d in dirs)


def test_set_seed_makes_torch_deterministic():
    torch = pytest.importorskip("torch")
    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    assert torch.equal(first, torch.rand(3))


def test_get_device_cpu_and_fallback():
    pytest.importorskip("torch")
    assert get_device("cpu") == "cpu"
    assert get_device("auto") in {"cpu", "cuda", "mps"}


def test_exception_context_in_message():
    err = InputValidationError("Text too long", context={"chars": 30000, "max": 20000})
    assert isinstance(err, SummarizerError)
    assert "chars=30000" in str(err)
    assert err.context["max"] == 20000


def test_exception_chaining_keeps_cause():
    with pytest.raises(ConfigurationError) as info:
        try:
            raise KeyError("model")
        except KeyError as cause:
            raise ConfigurationError("missing") from cause
    assert isinstance(info.value.__cause__, KeyError)


def test_logger_namespace_and_json_format():
    logger = get_logger("some.module")
    assert logger.name == "summarizer.some.module"

    record = logging.LogRecord("summarizer.x", logging.INFO, __file__, 1, "hi %s", ("there",), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hi there"
    assert payload["level"] == "INFO"
