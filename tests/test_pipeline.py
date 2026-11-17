"""Integration tests for the LLM proposal-generation pipeline.

Covers config validation, dataset transformation (raw -> SFT/DPO), and the
FastAPI endpoints (``/health``, ``/api/v1/generate-proposal``,
``/api/v1/format-dataset``). The suite runs fully offline: the inference engine
is pinned to the deterministic backend and ``cpu_demo`` execution mode.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import app
from src.config_loader import LoRAConfig, load_config
from src.dataset_builder import (
    build_dpo_records,
    build_reference_proposal,
    build_sft_records,
    compute_token_stats,
    load_jsonl,
    validate_job,
)

# Pin the runtime to the offline, CPU-safe path for deterministic CI runs.
os.environ.pop("LLM_API_KEY", None)
os.environ["EXECUTION_MODE"] = "cpu_demo"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def raw_jobs():
    return load_jsonl(DATA_DIR / "raw_jobs.jsonl")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_config_loads_and_validates():
    config = load_config()
    assert config.model.base_model.endswith("-bnb-4bit")
    assert config.model.max_seq_length == 2048
    assert config.lora.r == 16
    assert config.lora.lora_alpha == 32
    assert config.lora.lora_dropout == 0.05
    assert {"q_proj", "v_proj", "down_proj"} <= set(config.lora.target_modules)
    assert config.sft.learning_rate == 2.0e-4
    assert config.sft.num_train_epochs == 3
    assert config.dpo.beta == 0.1
    assert config.dpo.loss_type == "sigmoid"


def test_config_rejects_invalid_lora():
    with pytest.raises(ValidationError):
        LoRAConfig(target_modules=[], r=0)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #

def test_raw_jobs_are_valid(raw_jobs):
    assert len(raw_jobs) == 5
    for job in raw_jobs:
        assert validate_job(job)["id"].startswith("job-")


def test_build_reference_proposal_structure(raw_jobs):
    proposal = build_reference_proposal(raw_jobs[0])
    assert proposal.title
    assert proposal.executive_hook.lower().startswith("the core bottleneck")
    assert len(proposal.architecture) >= 1
    assert len(proposal.milestones) == 5
    assert proposal.timeline
    assert proposal.deliverables


def test_build_sft_records(raw_jobs):
    records = build_sft_records(raw_jobs)
    assert len(records) == len(raw_jobs)
    for record in records:
        assert record["instruction"]
        assert record["input"]
        assert record["output"].startswith("## Executive Hook")
        assert "## Proposed System Architecture" in record["output"]
        assert "## Milestone Execution Plan" in record["output"]
        assert record["text"]
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][-1]["role"] == "assistant"


def test_build_dpo_records(raw_jobs):
    preferences = load_jsonl(DATA_DIR / "dpo_preferences.jsonl")
    records = build_dpo_records(raw_jobs, preferences)
    assert len(records) == len(raw_jobs)
    for record in records:
        assert record["prompt"]
        assert record["chosen"]
        assert record["rejected"]
        assert record["prompt_tokens"] > 0
        # Chosen responses are concise vs. rejected fluff in the committed pairs.
        assert len(record["chosen"]) < len(record["rejected"])


def test_token_stats(raw_jobs):
    records = build_sft_records(raw_jobs)
    stats = compute_token_stats(records, ["output"])
    assert stats["output"]["samples"] == len(records)
    assert stats["output"]["min"] > 0
    assert stats["output"]["max"] >= stats["output"]["min"]


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model"]
    assert body["hardware_backend"] in {"gpu", "cpu_demo"}
    assert body["inference_backend"] in {"deterministic", "openai_compatible", "local"}
    assert body["version"]


def test_generate_proposal_endpoint(client):
    payload = {
        "title": "Test Job - Build a PostGIS API",
        "company": "TestCo",
        "description": "We need a spatial data API serving GeoJSON over PostGIS at scale.",
        "tech_stack": ["PostgreSQL", "PostGIS", "FastAPI"],
        "domain": "Geospatial",
        "budget": "$50k",
        "timeline": "8 weeks",
    }
    response = client.post("/api/v1/generate-proposal", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["backend"]
    assert body["latency_ms"] >= 0
    assert body["token_usage"]["total_tokens"] > 0
    proposal = body["proposal"]
    assert proposal["title"] == payload["title"]
    assert proposal["executive_hook"]
    assert proposal["milestones"]
    assert proposal["markdown"].startswith("## Executive Hook")


def test_generate_proposal_rejects_invalid(client):
    response = client.post("/api/v1/generate-proposal", json={"title": "Missing description"})
    assert response.status_code == 422


def test_format_dataset_endpoint(client):
    jobs = [
        {
            "title": "Test Geospatial Job",
            "description": "Serve municipal zoning data through GeoJSON endpoints.",
            "tech_stack": ["PostGIS", "AWS"],
            "domain": "geospatial",
            "timeline": "10 weeks",
        }
    ]
    response = client.post("/api/v1/format-dataset", json={"jobs": jobs, "include_dpo": True})
    assert response.status_code == 200
    body = response.json()
    assert len(body["sft_records"]) == 1
    assert len(body["dpo_records"]) == 1
    assert body["token_stats"]["output"]["samples"] == 1


def test_index_serves_prototype_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "prototype" in response.text.lower()
    assert "Generate proposal" in response.text