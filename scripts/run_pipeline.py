# ruff: noqa: E402  (repo-root import bootstrap must precede package imports)
"""Rich CLI orchestrator for the end-to-end proposal-generation pipeline.

Runs four stages sequentially and renders portfolio-grade terminal output:

1. Dataset ingestion & SFT/DPO schema validation.
2. Supervised Fine-Tuning (Unsloth 4-bit QLoRA / CPU dry-run).
3. Direct Preference Optimization (DPO alignment).
4. Context-aware technical proposal generation (formatted sample panel).

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config_loader import (
    EnvSettings,
    TrainingConfig,
    cuda_available,
    effective_model_name,
    load_config,
    load_settings,
    resolve_execution_mode,
)
from src.dataset_builder import (
    build_dpo_records,
    build_sft_records,
    compute_token_stats,
    load_jsonl,
    validate_job,
    write_jsonl,
)
from src.dpo_trainer import run_dpo
from src.inference_engine import InferenceEngine
from src.sft_trainer import run_sft

DATA_DIR = REPO_ROOT / "data"
console = Console()


def _stage_1(config: TrainingConfig, raw_jobs: List[Dict[str, Any]], preferences: List[Dict[str, Any]]) -> None:
    console.rule("[bold blue]Stage 1 — Dataset Ingestion & Schema Validation[/bold blue]")

    for job in raw_jobs:
        validate_job(job)

    sft_records = build_sft_records(raw_jobs)
    dpo_records = build_dpo_records(raw_jobs, preferences)

    dataset_dir = Path(config.output.dataset_dir)
    write_jsonl(dataset_dir / "sft_dataset.jsonl", sft_records)
    write_jsonl(dataset_dir / "dpo_preferences.jsonl", dpo_records)

    table = Table(title="Raw Job Postings (validated)")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Domain")
    table.add_column("Tech Stack")
    table.add_column("Timeline")
    for job in raw_jobs:
        table.add_row(
            job["id"],
            job["title"],
            job.get("domain", "-"),
            ", ".join(job.get("tech_stack", [])),
            job.get("timeline", "-"),
        )
    console.print(table)

    sft_stats = compute_token_stats(sft_records, ["instruction", "input", "output"])
    dpo_stats = compute_token_stats(dpo_records, ["prompt", "chosen", "rejected"])

    stats = Table(title="Token Length Statistics (heuristic)")
    stats.add_column("Field")
    stats.add_column("Samples", justify="right")
    stats.add_column("Min", justify="right")
    stats.add_column("Max", justify="right")
    stats.add_column("Mean", justify="right")
    for key, value in sft_stats.items():
        stats.add_row(f"SFT {key}", str(value["samples"]), str(value["min"]), str(value["max"]), str(value["mean"]))
    for key, value in dpo_stats.items():
        stats.add_row(f"DPO {key}", str(value["samples"]), str(value["min"]), str(value["max"]), str(value["mean"]))
    console.print(stats)

    console.print(
        f"[green]✓[/green] {len(raw_jobs)} jobs validated, {len(sft_records)} SFT records, "
        f"{len(dpo_records)} DPO pairs -> {dataset_dir}"
    )


def _stage_2(config: TrainingConfig, settings: EnvSettings, raw_jobs: List[Dict[str, Any]]) -> None:
    console.rule("[bold green]Stage 2 — Supervised Fine-Tuning (Unsloth 4-bit QLoRA)[/bold green]")
    metrics = run_sft(raw_jobs, config=config, settings=settings, console=console)
    if metrics.get("mode") == "cpu_demo":
        console.print(
            Panel(
                f"Backend: [cyan]{metrics['backend']}[/cyan]\n"
                f"Final loss: [bold]{metrics['final_loss']:.4f}[/bold]\n"
                f"Steps: {metrics['steps']}  |  Epochs: {metrics['num_train_epochs']}  |  "
                f"Dataset size: {metrics['dataset_size']}",
                title="SFT Summary",
            )
        )


def _stage_3(
    config: TrainingConfig,
    settings: EnvSettings,
    raw_jobs: List[Dict[str, Any]],
    preferences: List[Dict[str, Any]],
) -> None:
    console.rule("[bold magenta]Stage 3 — Direct Preference Optimization (DPO Alignment)[/bold magenta]")
    metrics = run_dpo(raw_jobs, preferences=preferences, config=config, settings=settings, console=console)
    if metrics.get("mode") == "cpu_demo":
        console.print(
            Panel(
                f"Beta: {metrics['beta']}  |  Loss type: [cyan]{metrics['loss_type']}[/cyan]\n"
                f"Final reward margin: [bold]{metrics['final_reward_margin']:.4f}[/bold]  |  "
                f"Dataset size: {metrics['dataset_size']}",
                title="DPO Summary",
            )
        )


def _stage_4(config: TrainingConfig, settings: EnvSettings, raw_jobs: List[Dict[str, Any]]) -> None:
    console.rule("[bold cyan]Stage 4 — Context-Aware Technical Proposal Generation[/bold cyan]")
    engine = InferenceEngine(config, settings)
    job = raw_jobs[0]
    result = engine.generate(job)
    console.print(
        Panel(
            result.proposal.render_markdown(),
            title=f"Sample Proposal — {job['title']}",
            subtitle=(
                f"backend={result.backend}  latency={result.latency_ms}ms  "
                f"tokens={result.token_usage['total_tokens']}  adapter={result.adapter_version}"
            ),
        )
    )


def main() -> None:
    console.print(
        Panel(
            "[bold]Automated LLM Fine-Tuning (SFT & DPO) & Technical Proposal Generation Pipeline[/bold]",
            subtitle="hardware-aware · Unsloth + TRL · FastAPI",
        )
    )

    settings = load_settings()
    config = load_config()
    mode = resolve_execution_mode(config, settings.execution_mode)
    hardware = "[bold green]CUDA GPU detected[/bold green]" if cuda_available() else "[bold yellow]No CUDA — CPU demo mode[/bold yellow]"

    console.print(
        Panel(
            f"Model: [cyan]{effective_model_name(config, settings)}[/cyan]\n"
            f"Execution mode: [bold]{mode}[/bold]\n"
            f"Hardware: {hardware}",
            title="Hardware-Aware Runtime",
        )
    )

    raw_jobs = load_jsonl(DATA_DIR / "raw_jobs.jsonl")
    preferences = load_jsonl(DATA_DIR / "dpo_preferences.jsonl")

    _stage_1(config, raw_jobs, preferences)
    _stage_2(config, settings, raw_jobs)
    _stage_3(config, settings, raw_jobs, preferences)
    _stage_4(config, settings, raw_jobs)

    console.print(
        Panel("[bold green]Pipeline complete[/bold green] — artifacts written to [cyan]outputs/[/cyan]", title="Done")
    )


if __name__ == "__main__":
    main()