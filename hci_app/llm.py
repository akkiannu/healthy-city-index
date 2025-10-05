"""Lightweight helper to generate planning suggestions via OpenAI."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

MODEL_NAME = "gpt-5-nano"


class LLMUnavailable(RuntimeError):
    """Raised when the language model cannot be reached or returns nothing."""


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable(
            "OPENAI_API_KEY is not set. Export it before requesting AI plans."
        )
    return OpenAI(api_key=api_key)


def _format_prompt(payload: Dict[str, Any]) -> str:
    """Build a concise prompt describing metrics and expectations for the LLM."""

    pretty_payload = json.dumps(payload, indent=2, ensure_ascii=False)
    instructions = (
        "Review the urban indicators below. "
        "For each pillar (Habitability, Parks & Greenery, Waste Management, Disease Risk)"
        " give a one-line insight, then list two very short actions and one data check."
    )
    return f"{instructions}\n\n{pretty_payload}"


def generate_plan(payload: Dict[str, Any]) -> str:
    """Call the OpenAI Responses API and return the generated advice."""

    prompt = _format_prompt(payload)
    client = _client()
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            instructions=(
                "You are an experienced urban planner. "
                "For each pillar give a single concise sentence, then provide two quick actions and one follow-up check."
            ),
        )
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        raise LLMUnavailable(f"LLM request failed: {exc}") from exc

    content = (response.output_text or "").strip()
    if not content:
        raise LLMUnavailable("Model returned an empty response. Try again later.")
    return content


__all__ = ["generate_plan", "LLMUnavailable"]
