"""FastAPI REST service for structured technical proposal generation.

Exposes:

* ``GET  /health``                        -> service/model/backend status.
* ``POST /api/v1/generate-proposal``      -> generate a proposal + latency/token usage.
* ``POST /api/v1/format-dataset``         -> format raw jobs into SFT/DPO JSONL payloads.

The app is dependency-light and boots offline: the inference engine falls back to
the deterministic template engine when no GPU or API key is available, so the API
can always be exercised end-to-end.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.config_loader import (
    EnvSettings,
    TrainingConfig,
    effective_model_name,
    load_config,
    load_settings,
    resolve_execution_mode,
)
from src.dataset_builder import build_dpo_records, build_sft_records, compute_token_stats
from src.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class JobSpec(BaseModel):
    """Pydantic-validated client job posting."""

    title: str
    company: str = ""
    description: str
    tech_stack: List[str] = Field(default_factory=list)
    domain: str = ""
    budget: str = ""
    timeline: str = ""
    location: str = ""
    experience_level: str = ""
    required_skills: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    id: Optional[str] = None


class TokenUsage(BaseModel):
    """Token accounting for a single generation call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ProposalResponse(BaseModel):
    """Response payload for the proposal generation endpoint."""

    proposal: Dict[str, Any]
    backend: str
    latency_ms: float
    token_usage: TokenUsage
    model: str
    lora_adapter_version: str


class FormatDatasetRequest(BaseModel):
    """Request payload for the dataset-formatting endpoint."""

    jobs: List[JobSpec]
    include_dpo: bool = True


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    model: str
    lora_adapter_version: str
    hardware_backend: str
    inference_backend: str
    version: str


app = FastAPI(
    title="LLM Proposal Generation API",
    version=__version__,
    description=(
        "Architecture-first technical proposal generation backed by an "
        "SFT + DPO aligned LoRA adapter and a hardware-aware inference engine."
    ),
)


@lru_cache(maxsize=1)
def get_config() -> TrainingConfig:
    """Load and cache the validated training config."""
    return load_config()


@lru_cache(maxsize=1)
def get_settings() -> EnvSettings:
    """Load and cache the environment settings."""
    return load_settings()


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    """Build and cache the inference engine."""
    return InferenceEngine(get_config(), get_settings())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service, model, adapter, and backend status."""
    config = get_config()
    settings = get_settings()
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        model=effective_model_name(config, settings),
        lora_adapter_version=engine.adapter_version(),
        hardware_backend=resolve_execution_mode(config, settings.execution_mode),
        inference_backend=engine.backend,
        version=__version__,
    )


@app.post("/api/v1/generate-proposal", response_model=ProposalResponse)
def generate_proposal(job: JobSpec) -> ProposalResponse:
    """Generate a structured technical proposal with latency and token telemetry."""
    try:
        result = get_engine().generate(job.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ProposalResponse(
        proposal=result.proposal.to_dict(),
        backend=result.backend,
        latency_ms=result.latency_ms,
        token_usage=TokenUsage(**result.token_usage),
        model=result.model,
        lora_adapter_version=result.adapter_version,
    )


@app.post("/api/v1/format-dataset")
def format_dataset(request: FormatDatasetRequest) -> Dict[str, Any]:
    """Validate raw job objects and return formatted SFT/DPO payloads."""
    try:
        jobs = [job.model_dump() for job in request.jobs]
        sft_records = build_sft_records(jobs)
        dpo_records = build_dpo_records(jobs) if request.include_dpo else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "sft_records": sft_records,
        "dpo_records": dpo_records,
        "token_stats": compute_token_stats(sft_records, ["instruction", "input", "output"]),
    }


# --------------------------------------------------------------------------- #
# Static web prototype
# --------------------------------------------------------------------------- #

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the single-page prototype UI."""
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))