"""Dataset construction for the SFT and DPO alignment stages.

Turns raw client job postings into instruction-tuning and preference-alignment
data:

* ``build_sft_records`` -> Alpaca ``{instruction, input, output}`` records whose
  outputs are the structured reference proposal synthesized from each job.
* ``build_dpo_records`` -> ``{prompt, chosen, rejected}`` preference pairs.
* ``compute_token_stats`` -> token-length telemetry used by the training dry-run.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a staff software architect who writes concise, architecture-first "
    "technical proposals that focus on the client's core bottleneck, a concrete "
    "system design, and a milestone delivery plan."
)

PROPOSAL_INSTRUCTION = (
    "You are a staff software architect. Write a concise, architecture-first "
    "technical proposal for the following client job posting. Address the core "
    "technical bottleneck, propose a concrete system architecture tailored to the "
    "requested stack, and lay out a milestone execution plan with timeline and "
    "deliverables."
)

_REQUIRED_JOB_FIELDS = ("title", "description", "tech_stack")


@dataclass
class Milestone:
    """A single delivery milestone in the execution plan."""

    name: str
    duration_weeks: int
    focus: str


@dataclass
class Proposal:
    """Structured technical proposal returned by the inference engine."""

    title: str
    executive_hook: str
    architecture: List[str]
    milestones: List[Milestone]
    timeline: str
    deliverables: List[str]
    raw_output: Optional[str] = None

    def render_markdown(self) -> str:
        """Render the proposal as Markdown (also used as the SFT target text)."""
        lines: List[str] = [
            "## Executive Hook",
            "",
            self.executive_hook,
            "",
            "## Proposed System Architecture",
            "",
        ]
        lines.extend(f"- {bullet}" for bullet in self.architecture)
        lines.extend(["", "## Milestone Execution Plan", ""])
        for index, milestone in enumerate(self.milestones, start=1):
            lines.append(
                f"{index}. **{milestone.name} ({milestone.duration_weeks}w)**: {milestone.focus}."
            )
        lines.extend(["", "## Estimated Timeline & Deliverables", ""])
        lines.append(f"**Timeline:** {self.timeline}")
        lines.append("**Deliverables:** " + ", ".join(self.deliverables) + ".")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return the serializable form used by the REST API."""
        payload = asdict(self)
        payload["markdown"] = self.render_markdown()
        return payload


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON-lines file, skipping blank/comment lines."""
    resolved = Path(path)
    records: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {resolved}: {exc}"
                ) from exc
    logger.info("loaded %d records from %s", len(records), resolved)
    return records


def write_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> Path:
    """Write records to a JSON-lines file, creating parent directories."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("wrote %d records to %s", len(records), resolved)
    return resolved


def clean_text(text: Any) -> str:
    """Normalize whitespace, unicode punctuation, and trailing punctuation spacing."""
    if not text:
        return ""
    value = str(text)
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value.strip()


def validate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a job posting carries the fields needed for synthesis."""
    if not isinstance(job, dict):
        raise ValueError(f"job entry must be a JSON object, got {type(job).__name__}")

    missing = [field for field in _REQUIRED_JOB_FIELDS if not job.get(field)]
    if missing:
        raise ValueError(f"job entry missing required fields: {', '.join(missing)}")

    if not isinstance(job.get("tech_stack"), list):
        raise ValueError(f"job '{job.get('title', '?')}' has a non-list tech_stack")

    return job


_ARCH_HINTS: Dict[str, str] = {
    "next.js": "Next.js App Router front end with React Server Components for instant, cacheable rendering.",
    "node.js": "TypeScript Node.js service layer exposing typed, versioned API contracts.",
    "postgresql": "PostgreSQL relational store with indexes tuned to the query access patterns.",
    "postgis": "PostGIS extension with GiST/SP-GiST spatial indexes for sub-second geometry queries.",
    "redis": "Redis cache layer for hot tiles, sessions, and realtime state.",
    "aws": "AWS infrastructure (Lambda/ECS/RDS) managed as reproducible infrastructure-as-code.",
    "lambda": "Serverless AWS Lambda compute with dead-letter queues for reliable event handling.",
    "kinesis": "Kinesis streams for durable, ordered event ingestion.",
    "step functions": "AWS Step Functions to orchestrate multi-stage event workflows.",
    "dynamodb": "DynamoDB for low-latency hot storage with on-demand scaling.",
    "s3": "S3 object store for cold data, queried downstream via Athena analytics.",
    "athena": "Athena SQL over S3 for cost-effective analytical queries.",
    "terraform": "Terraform for reproducible infrastructure and drift-free deploys.",
    "python": "Python service layer with Pydantic-validated schemas and async handlers.",
    "langgraph": "LangGraph agent orchestration with an explicit, typed state graph.",
    "pgvector": "pgvector for hybrid retrieval (semantic + keyword) over internal docs.",
    "fastapi": "FastAPI async endpoints with auto-generated OpenAPI contracts.",
    "hugging face": "Hugging Face Transformers/PEFT for open-weight serving and LoRA adapters.",
    "geoserver": "GeoServer for standards-compliant WMS/WFS tile serving.",
    "geojson": "GeoJSON API surface for interoperable spatial payloads.",
    "docker": "Containerized services with deterministic build artifacts.",
    "kubernetes": "Kubernetes for orchestration, autoscaling, and rolling deploys.",
    "trpc": "tRPC for end-to-end type safety between client and server.",
    "prisma": "Prisma ORM for type-safe database access and migrations.",
    "vercel": "Vercel edge deployment for the front-end layer.",
    "openai-compatible": "OpenAI-compatible model gateway for LLM inference with a clean swap path.",
}

_MILESTONE_TEMPLATES: List[tuple] = [
    ("Discovery & data model", "audit existing systems, lock the schema, and define success metrics"),
    ("Core data layer", "build the storage/indexing foundation and ingestion pipelines"),
    ("Service & API", "implement the backend service layer and typed contracts"),
    ("Experience & integration", "build the front end, agent, or tooling surface"),
    ("Hardening & launch", "load testing, CI/CD, observability, and production cutover"),
]

_DOMAIN_PATTERNS: List[tuple] = [
    (
        "geospatial",
        "spatial-query latency caused by unindexed ad-hoc SQL and no cache layer",
        "lock the PostGIS schema to the access patterns with GiST/SP-GiST indexes and a Redis cache, then serve tiles from a server-rendered Next.js layer",
    ),
    (
        "serverless",
        "the brittle batch ETL that drops events under load",
        "replace it with an event-driven pipeline backed by dead-letter queues and idempotent handlers",
    ),
    (
        "ai automation",
        "uncontrolled tool calls and output that drifts from the brand voice",
        "constrain the agent with a scoped function layer, RAG grounding, guardrails, and a LoRA fine-tuning loop",
    ),
    (
        "govtech",
        "spatial-query latency at municipal scale",
        "tune the PostGIS indexes to the zoning/flood/transport patterns and precompute tile pyramids",
    ),
    (
        "logistics",
        "stale tracking data and manual exception handling",
        "deliver sub-second updates over SSE/WebSocket and gate LLM-generated messages behind human approval",
    ),
]


def _architecture_bullets(stack: List[str]) -> List[str]:
    """Map stack items to architecture-level recommendations."""
    stack_lower = [str(item).lower() for item in stack]
    bullets: List[str] = []
    for key, hint in _ARCH_HINTS.items():
        if any(key in item for item in stack_lower):
            bullets.append(hint)
    if not bullets:
        bullets.append("Modular service architecture with clear API boundaries and observability.")
    return bullets[:8]


def _parse_weeks(timeline: str) -> Optional[int]:
    """Extract the leading integer week count from a timeline string."""
    match = re.search(r"(\d+)", str(timeline))
    return int(match.group(1)) if match else None


def _split_weeks(total: int) -> List[int]:
    """Split a total week budget across the five standard milestones."""
    proportions = [2, 3, 3, 3, 2]
    scale = total / sum(proportions)
    weeks = [max(1, round(proportion * scale)) for proportion in proportions]
    weeks[-1] += total - sum(weeks)
    return weeks


def _domain_pattern(domain: str, title: str, description: str) -> tuple:
    """Return the (bottleneck, solution) pair best matching the job's domain."""
    haystack = f"{domain} {title} {description}".lower()
    for keyword, bottleneck, solution in _DOMAIN_PATTERNS:
        if keyword in haystack:
            return bottleneck, solution
    return (
        "a mismatch between the requested stack and the existing system's architecture",
        "design a modular architecture that matches the requested stack and de-risk delivery with a milestone plan",
    )


def _derive_deliverables(job: Dict[str, Any]) -> List[str]:
    """Derive concrete deliverables from the client's success criteria."""
    deliverables = [str(item) for item in job.get("success_criteria", [])][:3]
    deliverables.append("documentation and CI/CD pipeline")
    deliverables.append("load-test and validation report")
    return deliverables


def build_reference_proposal(job: Dict[str, Any]) -> Proposal:
    """Synthesize a deterministic reference proposal for a validated job posting."""
    validate_job(job)
    title = str(job["title"])
    domain = str(job.get("domain", ""))
    description = str(job.get("description", ""))
    stack = job.get("tech_stack", [])
    timeline = str(job.get("timeline", "12 weeks"))
    total_weeks = _parse_weeks(timeline) or 12

    bottleneck, solution = _domain_pattern(domain, title, description)
    hook = f"The core bottleneck is {bottleneck}. {solution[0].upper() + solution[1:]}."

    durations = _split_weeks(total_weeks)
    milestones = [
        Milestone(name=name, duration_weeks=weeks, focus=focus)
        for (name, focus), weeks in zip(_MILESTONE_TEMPLATES, durations)
    ]

    return Proposal(
        title=title,
        executive_hook=hook,
        architecture=_architecture_bullets(stack),
        milestones=milestones,
        timeline=timeline,
        deliverables=_derive_deliverables(job),
    )


def build_job_summary(job: Dict[str, Any]) -> str:
    """Flatten a job posting into a compact text block used as the SFT/DPO prompt."""
    validate_job(job)
    stack = ", ".join(job.get("tech_stack", []))
    parts = [
        f"Client: {job.get('company', 'N/A')}.",
        f"Title: {job['title']}.",
        f"Description: {clean_text(job.get('description', ''))}",
        f"Tech stack: {stack}.",
        f"Budget: {job.get('budget', 'TBD')}.",
        f"Timeline: {job.get('timeline', 'TBD')}.",
    ]
    return " ".join(parts)


def estimate_tokens(text: Any) -> int:
    """Estimate token count without a tokenizer (word count vs. chars/4 heuristic)."""
    if not text:
        return 0
    value = str(text)
    words = len(re.findall(r"\S+", value))
    chars = len(value)
    return max(words, round(chars / 4))


def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Serialize a list of role/content messages into a ChatML-formatted string."""
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def build_sft_messages(instruction: str, input_text: str, output_text: str) -> List[Dict[str, str]]:
    """Assemble the system/user/assistant message triple for a single SFT sample."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": output_text},
    ]


def build_sft_records(raw_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform raw job postings into Alpaca/ChatML instruction records."""
    records: List[Dict[str, Any]] = []
    for job in raw_jobs:
        validate_job(job)
        proposal = build_reference_proposal(job)
        input_text = build_job_summary(job)
        output_text = proposal.render_markdown()
        messages = build_sft_messages(PROPOSAL_INSTRUCTION, input_text, output_text)
        records.append(
            {
                "instruction": PROPOSAL_INSTRUCTION,
                "input": input_text,
                "output": output_text,
                "messages": messages,
                "text": format_chatml(messages),
            }
        )
    logger.info("built %d SFT records", len(records))
    return records


_FLUFFY_REJECTED = (
    "In today's fast-paced digital landscape, we leverage best-in-class synergies "
    "to deliver a seamless, world-class solution that is robust, scalable, and "
    "future-proof. Our holistic, end-to-end approach harnesses state-of-the-art "
    "technologies to drive transformative outcomes, empower stakeholders, and "
    "unlock unprecedented value while maximizing ROI. We are fully committed to "
    "exceeding expectations and delivering an intuitive, delightful experience "
    "aligned with your long-term strategic vision."
)


def _normalize_preference(preference: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean a single DPO preference triple."""
    prompt = clean_text(preference.get("prompt", ""))
    chosen = clean_text(preference.get("chosen", ""))
    rejected = clean_text(preference.get("rejected", ""))
    if not all((prompt, chosen, rejected)):
        raise ValueError("DPO preference must include non-empty prompt, chosen, and rejected")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def build_dpo_records(
    raw_jobs: List[Dict[str, Any]],
    preferences: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build normalized DPO preference pairs, either from the committed pairs or synthesized."""
    if preferences:
        records = [_normalize_preference(preference) for preference in preferences]
    else:
        logger.info("no DPO preferences supplied; synthesizing pairs from raw jobs")
        records = []
        for job in raw_jobs:
            validate_job(job)
            chosen = build_reference_proposal(job).render_markdown()
            records.append(
                {
                    "prompt": f"{PROPOSAL_INSTRUCTION}\n\n{build_job_summary(job)}",
                    "chosen": chosen,
                    "rejected": _FLUFFY_REJECTED,
                }
            )

    for record in records:
        record["prompt_tokens"] = estimate_tokens(record["prompt"])
        record["chosen_tokens"] = estimate_tokens(record["chosen"])
        record["rejected_tokens"] = estimate_tokens(record["rejected"])

    logger.info("built %d DPO preference records", len(records))
    return records


def compute_token_stats(records: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    """Compute min/max/mean/total token statistics across the given record fields."""
    stats: Dict[str, Any] = {}
    for key in keys:
        lengths = [estimate_tokens(record.get(key, "")) for record in records]
        if lengths:
            stats[key] = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(statistics.mean(lengths), 2),
                "total": sum(lengths),
                "samples": len(lengths),
            }
    return stats


def to_hf_dataset(records: List[Dict[str, Any]]) -> Any:
    """Export records as a Hugging Face Dataset (lazy import of ``datasets``)."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HF Dataset export. "
            "Install it with: pip install -r requirements-ml.txt"
        ) from exc
    return Dataset.from_list(records)