"""Configuration loading and validation.

Single source of truth for hyperparameters, environment settings, and
hardware-aware execution-mode resolution. Validates the YAML registry with
Pydantic v2 and exposes optional overrides from ``.env`` via pydantic-settings.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "training_config.yaml"
DEFAULT_ENV_PATH: Path = REPO_ROOT / ".env"

ExecutionMode = Literal["auto", "gpu", "cpu_demo"]


class ModelConfig(BaseModel):
    """Base-model identifiers and sequence/packing settings."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_model: str
    max_seq_length: int = Field(default=2048, ge=64, le=32768)
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    attn_implementation: str = "flash_attention_2"
    use_gradient_checkpointing: Optional[str] = "unpaged_8bit"


class ExecutionConfig(BaseModel):
    """Runtime hardware-mode selection and reproducibility seed."""

    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode = "auto"
    device: str = "auto"
    seed: int = Field(default=42, ge=0)
    dry_run_steps: int = Field(default=12, ge=1, le=500)


class LoRAConfig(BaseModel):
    """QLoRA adapter tunables shared by the SFT and DPO stages."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    loftq_config: Optional[Dict[str, Any]] = None
    target_modules: List[str]
    modules_to_save: List[str] = Field(default_factory=lambda: ["lm_head", "embed_tokens"])

    @field_validator("target_modules")
    @classmethod
    def _target_modules_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("lora.target_modules must contain at least one module name")
        return value


class SFTConfig(BaseModel):
    """Supervised fine-tuning hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    trainer: Literal["unsloth", "hf"] = "unsloth"
    learning_rate: float = Field(default=2.0e-4, gt=0.0)
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_train_epochs: float = Field(default=3.0, gt=0.0)
    max_grad_norm: float = Field(default=0.3, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    optim: str = "adamw_8bit"
    logging_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=100, ge=1)
    packing: bool = False
    use_chat_template: bool = True


class DPOConfig(BaseModel):
    """Direct Preference Optimization alignment hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    beta: float = Field(default=0.1, gt=0.0)
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid"
    learning_rate: float = Field(default=5.0e-5, gt=0.0)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_train_epochs: float = Field(default=1.0, gt=0.0)
    max_length: int = Field(default=1024, ge=64)
    max_prompt_length: int = Field(default=512, ge=64)
    temperature: float = Field(default=0.1, ge=0.0)
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = Field(default=1, ge=1)


class InferenceConfig(BaseModel):
    """Decoding and backend-selection parameters for the proposal generator."""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.0)
    do_sample: bool = True
    backend: Literal["auto", "local", "openai_compatible", "deterministic"] = "auto"


class OutputConfig(BaseModel):
    """Artifact naming and output directories."""

    model_config = ConfigDict(extra="forbid")

    dir: str = "outputs"
    sft_lora_name: str = "lora-sft-adapter"
    dpo_lora_name: str = "lora-dpo-adapter"
    dataset_dir: str = "outputs/datasets"


class TrainingConfig(BaseModel):
    """Top-level registry aggregating every subsection above."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    lora: LoRAConfig
    sft: SFTConfig = Field(default_factory=SFTConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _validate_cross_section(self) -> "TrainingConfig":
        if self.dpo.max_length <= self.dpo.max_prompt_length:
            raise ValueError("dpo.max_length must be greater than dpo.max_prompt_length")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "TrainingConfig":
        """Load and validate the YAML hyperparameter registry."""
        resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not resolved.exists():
            raise FileNotFoundError(f"config file not found: {resolved}")

        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ValueError(f"config file {resolved} must contain a YAML mapping")

        logger.info("validated config from %s", resolved)
        return cls.model_validate(raw)


class EnvSettings(BaseSettings):
    """Optional environment overrides read from the process env and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(str(DEFAULT_ENV_PATH), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    model_name: Optional[str] = Field(default=None, alias="MODEL_NAME")
    execution_mode: Optional[str] = Field(default=None, alias="EXECUTION_MODE")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    llm_api_base: str = Field(default="https://openrouter.ai/api/v1", alias="LLM_API_BASE")
    llm_model: str = Field(default="meta-llama/llama-3.1-8b-instruct", alias="LLM_MODEL")


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Return True when a CUDA-capable PyTorch runtime is importable and visible."""
    try:
        import torch  # type: ignore  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - depends on the host environment
        return False


def resolve_execution_mode(
    config: TrainingConfig,
    env_execution_mode: Optional[str] = None,
) -> str:
    """Resolve the effective execution mode.

    Priority: explicit ``EXECUTION_MODE`` env var > YAML ``execution.mode``.
    ``gpu``/``auto`` map to real Unsloth training only when CUDA is available;
    otherwise we gracefully fall back to the CPU demo simulation.
    """
    requested = (env_execution_mode or config.execution.mode).strip().lower()

    if requested in ("gpu", "auto") and cuda_available():
        return "gpu"

    if requested == "gpu":
        logger.warning("EXECUTION_MODE=gpu requested but CUDA is unavailable; using cpu_demo")

    return "cpu_demo"


def effective_model_name(config: TrainingConfig, settings: Optional[EnvSettings] = None) -> str:
    """Return the resolved base-model identifier (env override wins)."""
    if settings is not None and settings.model_name:
        return settings.model_name
    return config.model.name


def load_config(path: Path | str | None = None) -> TrainingConfig:
    """Convenience wrapper around ``TrainingConfig.from_yaml``."""
    return TrainingConfig.from_yaml(path)


def load_settings() -> EnvSettings:
    """Load optional environment overrides."""
    return EnvSettings()