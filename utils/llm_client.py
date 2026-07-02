"""LiteLLM abstraction layer for swappable providers.

Supports:
- Groq
- Gemini
- Ollama
- Any LiteLLM-compatible provider

This version logs estimated request size and allows each agent to control
max_tokens so Groq does not exceed the 12k TPM/request budget.
"""

from __future__ import annotations

import json
import re
from typing import Any

import litellm
from loguru import logger

from utils.config_loader import load_config


def get_llm_config(override_model: str | None = None) -> dict[str, Any]:
    config = load_config()["llm"]

    if override_model:
        config = {**config, "model": override_model}

    return config


def rough_token_estimate(text: str) -> int:
    """Rough estimate: 1 token is usually around 4 characters."""
    return max(1, len(text) // 4)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    override_model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    cfg = get_llm_config(override_model)

    output_tokens = int(max_tokens if max_tokens is not None else cfg.get("max_tokens", 1600))
    temp = float(temperature if temperature is not None else cfg.get("temperature", 0.2))

    estimated_input_tokens = rough_token_estimate(system_prompt + user_prompt)
    estimated_total_tokens = estimated_input_tokens + output_tokens

    logger.info(
        "LLM request size estimate: "
        f"input_tokens≈{estimated_input_tokens}, "
        f"max_output_tokens={output_tokens}, "
        f"total≈{estimated_total_tokens}, "
        f"model={cfg['model']}"
    )

    try:
        response = litellm.completion(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            max_tokens=output_tokens,
            timeout=cfg.get("timeout", 120),
        )

        content = response.choices[0].message.content
        return content or ""

    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        raise


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return json.loads(match.group(0))

        raise