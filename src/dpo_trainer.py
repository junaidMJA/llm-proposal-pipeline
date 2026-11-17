"""Direct Preference Optimization (DPO) alignment engine.

Hardware-aware DPO module mirroring ``sft_trainer.py``:

* GPU mode runs the real TRL ``DPOTrainer`` on top of an Unsloth 4-bit LoRA model,
  using ``proxy/beta`` and ``loss_type=sigmoid``.
* CPU demo mode validates the chosen/rejected preference pairs and simulates the
  preference-optimization trajectory (rising reward margin, chosen-vs-rejected
  implicit rewards) before writing ``outputs/dpo_adapter_meta.json``.
"""

from __future__ import annotations

import json
import logging
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
from src.dataset_builder import SYSTEM_PROMPT, build_dpo_records, compute_token_stats

logger = logging.getLogger(__name__)


class DPOTrainerEngine:
    """Hardware-aware preference-alignment engine."""

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
        self.adapter_dir = self.output_dir / config.output.dpo_lora_name
        self.adapter_name = config.output.dpo_lora_name

    def run(
        self,
        raw_jobs: List[Dict[str, Any]],
        preferences: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the DPO dataset and execute the appropriate alignment path."""
        records = build_dpo_records(raw_jobs, preferences)
        token_stats = compute_token_stats(records, ["prompt", "chosen", "rejected"])

        if self.mode == "gpu":
            metrics = self._train_trl(records)
        else:
            metrics = self._simulate_alignment(records)

        metrics["token_stats"] = token_stats
        metrics["dataset_size"] = len(records)
        self._save_adapter_metadata(metrics)
        return metrics

    def _build_lora_config(self) -> Dict[str, Any]:
        """Build kwargs for the Unsloth ``get_peft_model`` call."""
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

    def _to_dpo_dataset(self, records: List[Dict[str, Any]]) -> Any:
        """Convert preference records into the TRL chat-template DPO format."""
        from datasets import Dataset  # type: ignore

        rows = []
        for record in records:
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": record["prompt"]},
                    ],
                    "chosen": [{"role": "assistant", "content": record["chosen"]}],
                    "rejected": [{"role": "assistant", "content": record["rejected"]}],
                }
            )
        return Dataset.from_list(rows)

    def _train_trl(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute real TRL DPOTrainer alignment on an Unsloth LoRA model (CUDA)."""
        try:
            from trl import DPOConfig, DPOTrainer
            from unsloth import FastLanguageModel, is_bfloat16_supported  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "GPU DPO requires the ML stack. Run: pip install -r requirements-ml.txt"
            ) from exc

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(model, **self._build_lora_config())

        dpo_args = DPOConfig(
            output_dir=str(self.adapter_dir),
            per_device_train_batch_size=self.config.dpo.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.dpo.gradient_accumulation_steps,
            learning_rate=self.config.dpo.learning_rate,
            num_train_epochs=self.config.dpo.num_train_epochs,
            lr_scheduler_type=self.config.dpo.lr_scheduler_type,
            optim=self.config.dpo.optim,
            logging_steps=self.config.dpo.logging_steps,
            max_length=self.config.dpo.max_length,
            max_prompt_length=self.config.dpo.max_prompt_length,
            seed=self.config.execution.seed,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            beta=self.config.dpo.beta,
            loss_type=self.config.dpo.loss_type,
            train_dataset=self._to_dpo_dataset(records),
            tokenizer=tokenizer,
        )

        result = trainer.train()
        trainer.save_model(str(self.adapter_dir))

        return {
            "mode": "gpu",
            "backend": "trl-dpo",
            "beta": self.config.dpo.beta,
            "loss_type": self.config.dpo.loss_type,
            "final_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
            "num_train_epochs": self.config.dpo.num_train_epochs,
        }

    def _simulate_alignment(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dry-run: simulate rising chosen-vs-rejected reward margins."""
        steps = self.config.execution.dry_run_steps
        beta = self.config.dpo.beta

        self.console.print(
            "[bright_black]DPO dry-run  |  mode=cpu_demo  backend=preference-sim  "
            f"steps={steps}  beta={beta}  loss_type={self.config.dpo.loss_type}[/bright_black]"
        )

        margins: List[float] = []
        chosen_rewards: List[float] = []
        rejected_rewards: List[float] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("[bold magenta]DPO alignment[/]", total=steps)
            for step in range(1, steps + 1):
                time.sleep(0.02)
                ratio = step / steps
                chosen = round(random.uniform(0.5, 1.5) + 0.4 * ratio, 4)
                margin = round(0.12 + 0.55 * ratio + random.uniform(-0.03, 0.03), 4)
                chosen_rewards.append(chosen)
                rejected_rewards.append(round(chosen - max(margin, 0.0), 4))
                margins.append(max(margin, 0.0))
                progress.update(task, advance=1)

        return {
            "mode": "cpu_demo",
            "backend": "dry-run-simulation",
            "beta": beta,
            "loss_type": self.config.dpo.loss_type,
            "final_reward_margin": margins[-1],
            "reward_margins": margins,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "num_train_epochs": self.config.dpo.num_train_epochs,
            "steps": steps,
        }

    def _save_adapter_metadata(self, metrics: Dict[str, Any]) -> Path:
        """Persist adapter metadata and a PEFT config for downstream consumers."""
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        (self.adapter_dir / "adapter_config.json").write_text(
            json.dumps(self._peft_adapter_config(), indent=2), encoding="utf-8"
        )

        metadata = {
            "stage": "dpo",
            "adapter_name": self.adapter_name,
            "base_model": self.model_name,
            "mode": metrics.get("mode"),
            "backend": metrics.get("backend"),
            "lora_config": self.config.lora.model_dump(),
            "training_hyperparameters": {
                "beta": self.config.dpo.beta,
                "loss_type": self.config.dpo.loss_type,
                "learning_rate": self.config.dpo.learning_rate,
                "num_train_epochs": self.config.dpo.num_train_epochs,
                "max_length": self.config.dpo.max_length,
                "max_prompt_length": self.config.dpo.max_prompt_length,
            },
            "metrics": metrics,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = self.output_dir / "dpo_adapter_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.console.print(f"[green]Saved adapter metadata -> {meta_path}[/green]")
        return meta_path


def run_dpo(
    raw_jobs: List[Dict[str, Any]],
    preferences: Optional[List[Dict[str, Any]]] = None,
    config: Optional[TrainingConfig] = None,
    settings: Optional[EnvSettings] = None,
    console: Optional[Console] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: build a DPO engine and run it end-to-end."""
    return DPOTrainerEngine(
        config or load_config(),
        settings=settings,
        console=console,
        output_dir=output_dir,
    ).run(raw_jobs, preferences=preferences)