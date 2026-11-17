"""Supervised Fine-Tuning (SFT) training engine.

Implements a hardware-aware SFT module:

* On CUDA boxes (``EXECUTION_MODE=gpu``) it runs real Unsloth 4-bit QLoRA training
  wrapped around the TRL ``SFTTrainer``.
* On CPU/macOS (``EXECUTION_MODE=cpu_demo``, the default fallback) it validates the
  dataset, instantiates the PEFT LoRA metadata, simulates the optimizer trajectory
  (loss decay + linear LR schedule) under a Rich progress bar, and emits a valid
  adapter metadata artifact to ``outputs/lora_adapter_meta.json``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.config_loader import (
    EnvSettings,
    TrainingConfig,
    effective_model_name,
    load_config,
    resolve_execution_mode,
)
from src.dataset_builder import build_sft_records, compute_token_stats

logger = logging.getLogger(__name__)


class SFTTrainerEngine:
    """Hardware-aware supervised fine-tuning engine."""

    def __init__(
        self,
        config: TrainingConfig,
        settings: Optional[EnvSettings] = None,
        console: Optional[Console] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.console = console or Console()
        self.model_name = effective_model_name(config, settings)
        self.mode = resolve_execution_mode(config, settings.execution_mode if settings else None)
        self.output_dir = Path(output_dir or config.output.dir)
        self.adapter_dir = self.output_dir / config.output.sft_lora_name
        self.adapter_name = config.output.sft_lora_name

    def run(self, raw_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the SFT dataset and execute the appropriate training path."""
        records = build_sft_records(raw_jobs)
        token_stats = compute_token_stats(records, ["instruction", "input", "output"])

        if self.mode == "gpu":
            metrics = self._train_unsloth(records)
        else:
            metrics = self._simulate_training(records)

        metrics["token_stats"] = token_stats
        metrics["dataset_size"] = len(records)
        self._save_adapter_metadata(metrics)
        return metrics

    def _build_lora_config(self) -> Dict[str, Any]:
        """Build the kwargs consumed by ``FastLanguageModel.get_peft_model``."""
        lora = self.config.lora
        return {
            "r": lora.r,
            "target_modules": lora.target_modules,
            "lora_alpha": lora.lora_alpha,
            "lora_dropout": lora.lora_dropout,
            "bias": lora.bias,
            "use_rslora": lora.use_rslora,
            "loftq_config": lora.loftq_config,
        }

    def _peft_adapter_config(self) -> Dict[str, Any]:
        """Render a PEFT-style ``adapter_config.json`` for artifact authenticity."""
        lora = self.config.lora
        return {
            "base_model_name_or_path": self.model_name,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": lora.r,
            "lora_alpha": lora.lora_alpha,
            "lora_dropout": lora.lora_dropout,
            "bias": lora.bias,
            "use_rslora": lora.use_rslora,
            "target_modules": lora.target_modules,
            "modules_to_save": lora.modules_to_save,
            "fan_in_fan_out": False,
        }

    def _decaying_loss(self, ratio: float) -> float:
        """Synthetic loss curve: exponential decay plus shrinking noise."""
        base = 0.85
        amplitude = 2.2
        decay = math.exp(-4.0 * ratio)
        noise = random.uniform(-0.03, 0.03) * (1.0 - ratio + 0.08)
        return round(base + amplitude * decay + noise, 4)

    def _train_unsloth(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute real Unsloth 4-bit QLoRA SFT (requires Linux + CUDA)."""
        try:
            from datasets import Dataset
            from transformers import TrainingArguments
            from trl import SFTTrainer
            from unsloth import FastLanguageModel, is_bfloat16_supported  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "GPU SFT requires the ML stack. Run: pip install -r requirements-ml.txt"
            ) from exc

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(model, **self._build_lora_config())

        training_args = TrainingArguments(
            output_dir=str(self.adapter_dir),
            per_device_train_batch_size=self.config.sft.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.sft.gradient_accumulation_steps,
            learning_rate=self.config.sft.learning_rate,
            num_train_epochs=self.config.sft.num_train_epochs,
            lr_scheduler_type=self.config.sft.lr_scheduler_type,
            warmup_ratio=self.config.sft.warmup_ratio,
            optim=self.config.sft.optim,
            weight_decay=self.config.sft.weight_decay,
            max_grad_norm=self.config.sft.max_grad_norm,
            logging_steps=self.config.sft.logging_steps,
            save_steps=self.config.sft.save_steps,
            seed=self.config.execution.seed,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=Dataset.from_list(records),
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            dataset_num_proc=1,
            packing=self.config.sft.packing,
            args=training_args,
        )

        result = trainer.train()
        trainer.save_model(str(self.adapter_dir))

        return {
            "mode": "gpu",
            "backend": "unsloth-qlora",
            "final_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
            "num_train_epochs": self.config.sft.num_train_epochs,
        }

    def _simulate_training(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dry-run: emit a realistic optimizer trajectory under a progress bar."""
        steps = self.config.execution.dry_run_steps
        learning_rate = self.config.sft.learning_rate
        effective_batch = (
            self.config.sft.per_device_train_batch_size
            * self.config.sft.gradient_accumulation_steps
        )

        self.console.print(
            "[bright_black]SFT dry-run  |  mode=cpu_demo  backend=peft-lora-sim  "
            f"steps={steps}  lr={learning_rate:.1e}  effective_batch={effective_batch}[/bright_black]"
        )

        loss_history: List[float] = []
        lr_history: List[float] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("[bold green]SFT alignment[/]", total=steps)
            for step in range(1, steps + 1):
                time.sleep(0.02)
                ratio = step / steps
                loss_history.append(self._decaying_loss(ratio))
                lr_history.append(round(learning_rate * (1.0 - ratio), 8))
                progress.update(task, advance=1)

        return {
            "mode": "cpu_demo",
            "backend": "dry-run-simulation",
            "final_loss": loss_history[-1],
            "loss_history": loss_history,
            "lr_history": lr_history,
            "num_train_epochs": self.config.sft.num_train_epochs,
            "steps": steps,
        }

    def _save_adapter_metadata(self, metrics: Dict[str, Any]) -> Path:
        """Persist adapter metadata and a PEFT config for downstream consumers."""
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        (self.adapter_dir / "adapter_config.json").write_text(
            json.dumps(self._peft_adapter_config(), indent=2), encoding="utf-8"
        )

        metadata = {
            "stage": "sft",
            "adapter_name": self.adapter_name,
            "base_model": self.model_name,
            "mode": metrics.get("mode"),
            "backend": metrics.get("backend"),
            "lora_config": self.config.lora.model_dump(),
            "training_hyperparameters": {
                "learning_rate": self.config.sft.learning_rate,
                "num_train_epochs": self.config.sft.num_train_epochs,
                "per_device_train_batch_size": self.config.sft.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.sft.gradient_accumulation_steps,
                "max_seq_length": self.config.model.max_seq_length,
            },
            "metrics": metrics,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = self.output_dir / "lora_adapter_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.console.print(f"[green]Saved adapter metadata -> {meta_path}[/green]")
        return meta_path


def run_sft(
    raw_jobs: List[Dict[str, Any]],
    config: Optional[TrainingConfig] = None,
    settings: Optional[EnvSettings] = None,
    console: Optional[Console] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: build an SFT engine and run it end-to-end."""
    return SFTTrainerEngine(
        config or load_config(),
        settings=settings,
        console=console,
        output_dir=output_dir,
    ).run(raw_jobs)