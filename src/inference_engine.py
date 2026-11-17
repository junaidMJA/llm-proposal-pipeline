"""Proposal inference engine.

Generates structured technical proposals from raw job postings through one of
three backends, resolved in hardware/credential-aware order:

1. ``local``            -> local Transformers pipeline when LoRA weights exist.
2. ``openai_compatible``-> hosted OpenAI-compatible endpoint (OpenRouter etc.) when
                           ``LLM_API_KEY`` is configured in ``.env``.
3. ``deterministic``    -> offline template-synthesis fallback so the pipeline is
                           always runnable with zero network or GPU dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.config_loader import EnvSettings, TrainingConfig, effective_model_name, load_config
from src.dataset_builder import (
    PROPOSAL_INSTRUCTION,
    SYSTEM_PROMPT,
    Milestone,
    Proposal,
    build_job_summary,
    build_reference_proposal,
    estimate_tokens,
    validate_job,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Output of a single proposal generation call plus telemetry."""

    proposal: Proposal
    backend: str
    latency_ms: float
    token_usage: Dict[str, int]
    model: str
    adapter_version: str
    raw_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the serializable form consumed by the REST API."""
        return {
            "proposal": self.proposal.to_dict(),
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "model": self.model,
            "lora_adapter_version": self.adapter_version,
        }


def _extract_section(text: str, header: str) -> str:
    """Extract the body of a Markdown ``## header`` section from raw model output."""
    pattern = rf"##\s*{re.escape(header)}[^\n]*\n(.*?)(?=##\s|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_bullets(text: str, header: str) -> List[str]:
    """Extract bullet points from a Markdown section."""
    section = _extract_section(text, header)
    return [
        re.sub(r"^\s*[-*]\s*", "", line).strip()
        for line in section.splitlines()
        if line.strip().startswith(("-", "*"))
    ]


def _extract_milestones(text: str, header: str) -> List[Milestone]:
    """Extract numbered milestones of the form ``1. **Name (Nw)**: focus.``"""
    section = _extract_section(text, header)
    milestones: List[Milestone] = []
    for line in section.splitlines():
        match = re.match(r"\s*\d+\.\s*\*\*(.+?)\s*\((\d+)w\)\*\*:\s*(.*)", line)
        if match:
            milestones.append(
                Milestone(
                    name=match.group(1).strip(),
                    duration_weeks=int(match.group(2)),
                    focus=match.group(3).strip().rstrip("."),
                )
            )
    return milestones


def _extract_timeline(text: str) -> str:
    """Extract the ``**Timeline:**`` value from the deliverables section."""
    section = _extract_section(text, "Estimated Timeline")
    match = re.search(r"Timeline:\s*\**\s*(.*)", section)
    return match.group(1).strip() if match else ""


def _extract_deliverables(text: str) -> List[str]:
    """Extract comma-separated deliverables from the timeline section."""
    section = _extract_section(text, "Estimated Timeline")
    match = re.search(r"Deliverables:\s*\**\s*(.*)", section)
    if not match:
        return []
    return [item.strip().rstrip(",") for item in match.group(1).split(",") if item.strip()]


def _first_line(text: str) -> str:
    """Return the first non-empty line of a block of text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_markdown_proposal(text: str, title: str) -> Proposal:
    """Best-effort parse of raw LLM output into a structured ``Proposal``."""
    return Proposal(
        title=title,
        executive_hook=_extract_section(text, "Executive Hook") or _first_line(text),
        architecture=_extract_bullets(text, "Proposed System Architecture"),
        milestones=_extract_milestones(text, "Milestone Execution Plan"),
        timeline=_extract_timeline(text) or "TBD",
        deliverables=_extract_deliverables(text),
        raw_output=text,
    )


class InferenceEngine:
    """Hardware/credential-aware proposal generator."""

    def __init__(self, config: TrainingConfig, settings: Optional[EnvSettings] = None) -> None:
        self.config = config
        self.settings = settings
        self.model_name = effective_model_name(config, settings)
        self.backend = self._resolve_backend()
        logger.info("inference backend resolved to %r for model %s", self.backend, self.model_name)

    def _resolve_backend(self) -> str:
        """Resolve the inference backend honoring config overrides and credentials."""
        requested = self.config.inference.backend
        if requested in ("local", "deterministic"):
            return requested
        if requested == "openai_compatible":
            return (
                "openai_compatible"
                if (self.settings and self.settings.llm_api_key)
                else "deterministic"
            )
        # auto mode: prefer a hosted LLM, otherwise stay fully offline.
        if self.settings and self.settings.llm_api_key:
            return "openai_compatible"
        return "deterministic"

    def adapter_version(self) -> str:
        """Return the current LoRA adapter version from saved artifacts."""
        meta_path = Path(self.config.output.dir) / "lora_adapter_meta.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                name = data.get("adapter_name", self.config.output.sft_lora_name)
                saved = (data.get("saved_at") or "")[:10]
                return f"{name}-{saved or 'v1.0.0'}"
            except (json.JSONDecodeError, OSError):
                pass
        return f"{self.config.output.sft_lora_name}@1.0.0"

    def generate(self, job: Dict[str, Any]) -> GenerationResult:
        """Generate a proposal for a validated job posting and time the call."""
        validate_job(job)
        start = time.perf_counter()
        proposal, token_usage, raw_output = self._dispatch(job)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return GenerationResult(
            proposal=proposal,
            backend=self.backend,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model=self.model_name,
            adapter_version=self.adapter_version(),
            raw_output=raw_output,
        )

    def _dispatch(self, job: Dict[str, Any]) -> Tuple[Proposal, Dict[str, int], str]:
        """Route to the active backend, falling back to deterministic on errors."""
        if self.backend == "openai_compatible":
            return self._generate_openai(job)
        if self.backend == "local":
            try:
                return self._generate_local(job)
            except Exception as exc:  # pragma: no cover - requires local weights
                logger.warning("local backend failed (%s); using deterministic", exc)
        return self._generate_deterministic(job)

    def _generate_deterministic(self, job: Dict[str, Any]) -> Tuple[Proposal, Dict[str, int], str]:
        """Offline template-synthesis fallback (always available)."""
        proposal = build_reference_proposal(job)
        raw_output = proposal.render_markdown()
        prompt = f"{PROPOSAL_INSTRUCTION}\n\n{build_job_summary(job)}"
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(raw_output)
        proposal.raw_output = raw_output
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return proposal, token_usage, raw_output

    def _generate_openai(self, job: Dict[str, Any]) -> Tuple[Proposal, Dict[str, int], str]:
        """Call an OpenAI-compatible chat-completions endpoint."""
        prompt = f"{PROPOSAL_INSTRUCTION}\n\n{build_job_summary(job)}"
        base = (self.settings.llm_api_base or "").rstrip("/")
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.inference.max_new_tokens,
            "temperature": self.config.inference.temperature,
            "top_p": self.config.inference.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120) as client:
            response = client.post(f"{base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_output = data["choices"][0]["message"]["content"]
        proposal = parse_markdown_proposal(raw_output, str(job["title"]))
        usage = data.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", estimate_tokens(prompt))),
            "completion_tokens": int(usage.get("completion_tokens", estimate_tokens(raw_output))),
            "total_tokens": int(
                usage.get("total_tokens", estimate_tokens(prompt) + estimate_tokens(raw_output))
            ),
        }
        return proposal, token_usage, raw_output

    def _generate_local(self, job: Dict[str, Any]) -> Tuple[Proposal, Dict[str, int], str]:
        """Run a local Transformers text-generation pipeline (weights required)."""
        from transformers import pipeline  # type: ignore

        prompt = f"{PROPOSAL_INSTRUCTION}\n\n{build_job_summary(job)}"
        pipe = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        outputs = pipe(
            prompt,
            max_new_tokens=self.config.inference.max_new_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            do_sample=self.config.inference.do_sample,
        )
        generated = outputs[0]["generated_text"]
        content = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
        proposal = parse_markdown_proposal(content or generated, str(job["title"]))
        token_usage = {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": estimate_tokens(content),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(content),
        }
        return proposal, token_usage, content


def generate_proposal(
    job: Dict[str, Any],
    config: Optional[TrainingConfig] = None,
    settings: Optional[EnvSettings] = None,
) -> GenerationResult:
    """Convenience wrapper: generate a proposal with a single call."""
    return InferenceEngine(config or load_config(), settings=settings).generate(job)