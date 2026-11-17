"""Automated LLM Fine-Tuning (SFT & DPO) & Technical Proposal Generation Pipeline.

This package implements a two-stage alignment loop:

1. **Supervised Fine-Tuning (SFT)** — teaches the model the structured shape of a
   high-signal technical proposal (hook, architecture, milestones, timeline).
2. **Direct Preference Optimization (DPO)** — aligns tone toward concise,
   architectural, ROI-focused responses over generic "ChatGPT-flavoured" prose.

The pipeline is hardware-aware: on GPU boxes it executes real Unsloth 4-bit QLoRA
training; anywhere else it runs a faithful telemetry simulation so the system can
be exercised end-to-end without CUDA.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]