"""LLM helper for generating urban planning suggestions."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Optional

from openai import OpenAI

# Cheapest hosted tier: GPT-4.1 Nano delivers low-latency, low-cost planning hints.
MODEL_NAME = "gpt-5-nano"


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot be contacted."""


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable(
            "OPENAI_API_KEY environment variable is not set. "
            "Provide a key before requesting AI recommendations."
        )
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=64)
def generate_planning_advice(payload_hash: str, prompt: str) -> str:
    """Call the OpenAI Responses API and return the generated advice."""

    client = _client()
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            instructions=(
                "You are an urban planning strategist. "
                "Analyse the provided city metrics and deliver a concise, actionable plan. "
                "Prioritise the top risks, mention specific interventions, and end with one data follow-up."
            ),
        )
    except Exception as exc:  # pragma: no cover - depends on network/LLM availability
        raise LLMUnavailable(f"LLM request failed: {exc}") from exc

    content = response.output_text or ""
    return content.strip()


def build_prompt(payload: Dict) -> str:
    """Format the metrics payload into a readable prompt."""

    pretty_payload = json.dumps(payload, indent=2, ensure_ascii=False)
    guidance = (
        "Using the metrics below, outline up to three priority actions for urban planners. "
        "Reference the key indicators driving each action. Close with one suggested data follow-up.\n\n"
    )
    return guidance + pretty_payload


def llm_recommendations(payload: Dict) -> str:
    """Generate planning advice for the given payload."""

    prompt = build_prompt(payload)
    cache_key = json.dumps(payload, sort_keys=True)
    return generate_planning_advice(cache_key, prompt)


__all__ = ["llm_recommendations", "LLMUnavailable"]
