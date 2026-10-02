"""API tests with FastAPI's TestClient (no network, no real server).

The app is built with ``create_app(pipeline=...)`` so most tests use a fake
model; the ``slow`` test at the end serves the real tiny PEGASUS model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from summarizer.config import AppSettings, ConfigurationManager
from summarizer.exception import PredictionError
from summarizer.pipeline.prediction_pipeline import PredictionPipeline
from tests.test_prediction_pipeline import DIALOGUE, TINY, FakeGenerator


def make_client(real_config_paths, generator=None, raise_errors=True, **settings_overrides):
    settings = AppSettings(
        _env_file=None,
        model_source="path",
        model_name_or_path="test/model",
        cors_origins="https://example.dev",
        **settings_overrides,
    )
    config = ConfigurationManager(*real_config_paths, smoke_test=False)
    pipeline = PredictionPipeline(settings, config=config, generator=generator or FakeGenerator())
    app = create_app(settings=settings, pipeline=pipeline)
    return TestClient(app, raise_server_exceptions=raise_errors)


@pytest.fixture
def client(real_config_paths):
    with make_client(real_config_paths) as c:  # `with` runs the lifespan (start-up)
        yield c


# ----------------------------------------------------------- info routes
def test_root(client):
    body = client.get("/").json()
    assert body["model"] == "test/model" and body["model_loaded"] is True
    assert body["endpoints"]["summarize"] == "POST /summarize"


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and response.json()["device"] == "cpu"


def test_health_503_when_model_failed_to_load(real_config_paths):
    class BrokenGenerator(FakeGenerator):
        def load(self):
            from summarizer.exception import ModelNotReadyError

            raise ModelNotReadyError("weights missing")

    with make_client(real_config_paths, generator=BrokenGenerator(loaded=False)) as c:
        health = c.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "unavailable"
        assert "weights missing" in health.json()["load_error"]

        response = c.post("/summarize", json={"text": DIALOGUE})
        assert response.status_code == 503
        assert response.json()["error"] == "model_not_ready"


def test_config_endpoint(client):
    assert client.get("/config").json()["max_batch_size"] == 8


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/", "/health", "/summarize", "/summarize/batch"} <= set(paths)


def test_frontend_is_served(client):
    page = client.get("/ui/")
    assert page.status_code == 200 and "AI Content Summarizer" in page.text
    assert client.get("/ui/app.js").status_code == 200


# -------------------------------------------------------------- summarize
def test_summarize_success(client):
    response = client.post("/summarize", json={"text": DIALOGUE})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Anna and Bob meet at noon."
    assert body["original_length"] == len(DIALOGUE)
    assert body["compression_ratio"] == pytest.approx(
        len(DIALOGUE) / len(body["summary"]), abs=0.01
    )
    assert set(body) >= {
        "summary",
        "original_length",
        "summary_length",
        "compression_ratio",
        "processing_time_seconds",
    }


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        ({"text": ""}, "validation_error", "body.text"),  # Pydantic: min_length=1
        ({"text": "    "}, "invalid_input", "empty"),
        ({"text": "too short"}, "invalid_input", "too short"),
        ({}, "validation_error", "Field required"),
        ({"text": 123}, "validation_error", "string"),
    ],
)
def test_summarize_invalid_input_is_422(client, payload, error, message):
    response = client.post("/summarize", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == error
    assert message in response.json()["detail"]
    assert response.json()["request_id"]


def test_very_long_text_is_422(real_config_paths):
    with make_client(real_config_paths, max_input_chars=500) as c:
        response = c.post("/summarize", json={"text": "word " * 200})
    assert response.status_code == 422 and "too long" in response.json()["detail"]


def test_malformed_json_is_422(client):
    response = client.post(
        "/summarize", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_get_on_summarize_is_405(client):
    assert client.get("/summarize").status_code == 405


def test_prediction_error_is_500_without_internals(real_config_paths):
    class FailingGenerator(FakeGenerator):
        def generate(self, texts, params, batch_size=8):
            raise PredictionError("CUDA out of memory at layer 7")

    with make_client(real_config_paths, generator=FailingGenerator()) as c:
        response = c.post("/summarize", json={"text": DIALOGUE})
    assert response.status_code == 500
    assert response.json()["error"] == "prediction_failed"
    assert "CUDA" not in response.json()["detail"]


def test_unexpected_error_is_generic_500(real_config_paths):
    class CrashingGenerator(FakeGenerator):
        def count_tokens(self, text):
            raise RuntimeError("secret internal detail")

    with make_client(real_config_paths, generator=CrashingGenerator(), raise_errors=False) as c:
        response = c.post("/summarize", json={"text": DIALOGUE})
    assert response.status_code == 500
    assert "secret" not in response.text


# ------------------------------------------------------------------ batch
def test_batch_success(client):
    response = client.post("/summarize/batch", json={"texts": [DIALOGUE, DIALOGUE]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2 and len(body["results"]) == 2
    assert body["total_processing_time_seconds"] >= 0


@pytest.mark.parametrize(
    ("texts", "message"),
    [([], "at least 1 item"), ([DIALOGUE, "bad"], "texts[1]"), ([DIALOGUE] * 9, "Too many texts")],
)
def test_batch_invalid_is_422(client, texts, message):
    response = client.post("/summarize/batch", json={"texts": texts})
    assert response.status_code == 422
    assert message in response.json()["detail"]


# ------------------------------------------------------------ middleware
def test_request_id_and_timing_headers(client):
    response = client.get("/", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert float(response.headers["X-Process-Time"]) >= 0
    assert len(client.get("/").headers["X-Request-ID"]) == 12  # generated when absent


def test_cors_allows_only_configured_origin(client):
    allowed = client.options(
        "/summarize",
        headers={"Origin": "https://example.dev", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "https://example.dev"
    blocked = client.options(
        "/summarize",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in blocked.headers


# --------------------------------------------------------- real tiny model
@pytest.mark.slow
def test_full_stack_with_tiny_model(real_config_paths):
    settings = AppSettings(
        _env_file=None, model_source="path", model_name_or_path=TINY, device="cpu"
    )
    with TestClient(create_app(settings=settings)) as c:  # loads the model at start-up
        assert c.get("/health").status_code == 200
        response = c.post("/summarize", json={"text": DIALOGUE})
        assert response.status_code == 200 and response.json()["input_tokens"] > 0
