"""FastAPI application: REST API + static web UI.

Run locally::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
    # Swagger UI: http://localhost:8000/docs   Web UI: http://localhost:8000/ui/

Endpoints

=====================  ======  ===============================================
``/``                  GET     Service information and links
``/health``            GET     Readiness: 200 if the model is loaded, else 503
``/summarize``         POST    Summarise one text
``/summarize/batch``   POST    Summarise up to MAX_BATCH_SIZE texts
``/ui/``               GET     Web interface (static files in ``frontend/``)
``/docs``, ``/redoc``  GET     Interactive API documentation
=====================  ======  ===============================================

Error mapping (custom exceptions -> HTTP status codes)

* ``InputValidationError`` (empty / too short / too long / too many) -> **422**
* ``ModelNotReadyError`` (model failed to load or is still loading)  -> **503**
* ``PredictionError`` and anything unexpected                        -> **500**
  (internal details are logged, never sent to the client)

The summarise endpoints are plain ``def`` (not ``async def``). FastAPI runs them
in a worker thread, so a multi-second model call never blocks the event loop
that answers ``/health``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    BatchSummarizeRequest,
    BatchSummarizeResponse,
    ErrorResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from summarizer import __version__
from summarizer.config.settings import AppSettings, get_settings
from summarizer.constants import PROJECT_ROOT
from summarizer.exception import (
    InputValidationError,
    ModelNotReadyError,
    PredictionError,
    SummarizerError,
)
from summarizer.logger import get_logger
from summarizer.pipeline.prediction_pipeline import PredictionPipeline

logger = get_logger("api")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Invalid input"},
    500: {"model": ErrorResponse, "description": "Generation failed"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


def _error(request: Request, status: int, code: str, detail: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorResponse(error=code, detail=detail, request_id=request_id)
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(
    settings: AppSettings | None = None, pipeline: PredictionPipeline | None = None
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: Runtime settings (default: from environment / .env).
        pipeline: Pre-built pipeline. Tests inject one with a fake model;
            normally it is created and loaded at start-up.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.started_at = time.time()
        app.state.pipeline = pipeline
        if app.state.pipeline is None:
            try:
                app.state.pipeline = PredictionPipeline(settings)
            except SummarizerError as err:  # bad configuration: refuse to start
                logger.error("Invalid configuration: %s", err)
                raise
        if settings.load_model_on_startup and not app.state.pipeline.is_ready:
            try:
                app.state.pipeline.load()
            except ModelNotReadyError as err:
                # Keep serving so /health can report the problem (returns 503).
                logger.error("Model failed to load: %s", err)
        yield
        logger.info("Shutting down")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Abstractive summarization of dialogues and long-form text with PEGASUS "
            "fine-tuned on SAMSum. Try it in the **Try it out** panels below, or use "
            "the web UI at [/ui/](/ui/)."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id and log method, path, status and duration."""
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        if request.url.path not in ("/health",):  # health checks would flood the logs
            logger.info(
                "%s %s -> %d (%.3fs) [%s]",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
                request.state.request_id,
            )
        return response

    # ------------------------------------------------------ error handlers
    @app.exception_handler(RequestValidationError)
    async def malformed_request(request: Request, err: RequestValidationError):
        # Same error shape as our own errors, e.g. "body.text: Field required".
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in err.errors()
        )
        return _error(request, 422, "validation_error", problems)

    @app.exception_handler(InputValidationError)
    async def invalid_input(request: Request, err: InputValidationError):
        return _error(request, 422, "invalid_input", err.message)

    @app.exception_handler(ModelNotReadyError)
    async def not_ready(request: Request, err: ModelNotReadyError):
        return _error(
            request, 503, "model_not_ready", "The model is not loaded yet. Try again shortly."
        )

    @app.exception_handler(PredictionError)
    async def prediction_failed(request: Request, err: PredictionError):
        logger.error("Prediction failed [%s]: %s", request.state.request_id, err)
        return _error(request, 500, "prediction_failed", "The model failed to generate a summary.")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, err: Exception):
        logger.exception("Unhandled error [%s]", getattr(request.state, "request_id", "-"))
        return _error(request, 500, "internal_error", "Unexpected server error.")

    def get_pipeline(request: Request) -> PredictionPipeline:
        pipe = getattr(request.app.state, "pipeline", None)
        if pipe is None:
            raise ModelNotReadyError("Pipeline not initialised")
        return pipe

    # -------------------------------------------------------------- routes
    @app.get("/", tags=["info"], summary="Service information")
    def root(request: Request) -> dict:
        pipe = getattr(request.app.state, "pipeline", None)
        return {
            "name": settings.app_name,
            "version": __version__,
            "description": "Abstractive dialogue summarization (PEGASUS fine-tuned on SAMSum)",
            "model": pipe.model_version if pipe else None,
            "model_loaded": bool(pipe and pipe.is_ready),
            "endpoints": {
                "health": "/health",
                "summarize": "POST /summarize",
                "batch": "POST /summarize/batch",
                "docs": "/docs",
                "ui": "/ui/",
            },
        }

    @app.get(
        "/health",
        tags=["info"],
        summary="Readiness check",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "Model not loaded"}},
    )
    def health(request: Request):
        pipe = getattr(request.app.state, "pipeline", None)
        info = pipe.info() if pipe else {}
        ready = bool(pipe and pipe.is_ready)
        body = HealthResponse(
            status="ok" if ready else "unavailable",
            model_loaded=ready,
            model=info.get("model"),
            device=info.get("device"),
            generation_profile=info.get("generation_profile"),
            load_seconds=info.get("load_seconds"),
            load_error=info.get("load_error"),
            uptime_seconds=round(time.time() - request.app.state.started_at, 1),
            version=__version__,
        )
        return JSONResponse(status_code=200 if ready else 503, content=body.model_dump())

    @app.get("/config", tags=["info"], summary="Limits and generation settings (used by the UI)")
    def public_config(request: Request) -> dict:
        return get_pipeline(request).info()

    @app.post(
        "/summarize",
        tags=["summarization"],
        summary="Summarise one text",
        response_model=SummarizeResponse,
        responses=ERROR_RESPONSES,
    )
    def summarize(body: SummarizeRequest, request: Request) -> SummarizeResponse:
        result = get_pipeline(request).summarize(body.text)
        return SummarizeResponse(**result.to_dict())

    @app.post(
        "/summarize/batch",
        tags=["summarization"],
        summary="Summarise several texts in one call",
        response_model=BatchSummarizeResponse,
        responses=ERROR_RESPONSES,
    )
    def summarize_batch(body: BatchSummarizeRequest, request: Request) -> BatchSummarizeResponse:
        results, total = get_pipeline(request).summarize_batch(body.texts)
        return BatchSummarizeResponse(
            results=[SummarizeResponse(**r.to_dict()) for r in results],
            count=len(results),
            total_processing_time_seconds=total,
        )

    if FRONTEND_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=Path(FRONTEND_DIR), html=True), name="ui")
    return app


app = create_app()
