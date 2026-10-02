"""Request and response models for the REST API.

Pydantic validates every request body against these classes *before* our
code runs: a missing field, wrong type or empty string is rejected with HTTP
422 and a precise error message. The same classes generate the OpenAPI
schema shown at ``/docs``.

Only structural rules live here. Limits you may want to change per
deployment (max characters, min words, batch size) come from environment
variables and are enforced by the PredictionPipeline, so changing them never
requires a code change.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_DIALOGUE = (
    "Amanda: I baked cookies. Do you want some?\n"
    "Jerry: Sure! What kind?\n"
    "Amanda: Chocolate chip. I'll bring you some tomorrow :-)\n"
    "Jerry: Great, thanks!"
)


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": EXAMPLE_DIALOGUE}]})

    text: str = Field(
        ...,
        min_length=1,
        description="Dialogue or long-form text to summarise. Chat format "
        "('Name: message' per line) gives the best results.",
    )


class BatchSummarizeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"texts": [EXAMPLE_DIALOGUE, EXAMPLE_DIALOGUE]}]}
    )

    texts: list[str] = Field(..., min_length=1, description="Texts to summarise in one batch.")


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Generated abstractive summary.")
    original_length: int = Field(..., description="Input length in characters.")
    summary_length: int = Field(..., description="Summary length in characters.")
    compression_ratio: float = Field(..., description="original_length / summary_length.")
    processing_time_seconds: float = Field(..., description="Model time for this request.")
    original_word_count: int
    summary_word_count: int
    input_tokens: int = Field(..., description="Tokens the model saw before truncation.")
    truncated: bool = Field(..., description="True if the input exceeded the model's token limit.")
    model_version: str


class BatchSummarizeResponse(BaseModel):
    results: list[SummarizeResponse]
    count: int
    total_processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the model is loaded, else 'unavailable'.")
    model_loaded: bool
    model: str | None = None
    device: str | None = None
    generation_profile: str | None = None
    load_seconds: float | None = None
    load_error: str | None = None
    uptime_seconds: float
    version: str


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")
    request_id: str | None = None
