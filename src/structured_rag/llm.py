from __future__ import annotations

import os
from typing import Any, Dict

import requests


def _ollama_chat(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 180,
    max_tokens: int = 400,
) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    message = data.get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("Ollama response did not include message.content")
    return str(content)


def _groq_chat(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 180,
    max_tokens: int = 400,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for --llm groq")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected Groq response format") from exc


def generate_answer(
    prompt: str,
    llm: str,
    model: str,
    temperature: float = 0.0,
    timeout: int = 180,
    max_tokens: int = 400,
) -> str:
    if llm == "ollama":
        return _ollama_chat(
            model=model,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    if llm == "groq":
        return _groq_chat(
            model=model,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported llm backend: {llm}")
