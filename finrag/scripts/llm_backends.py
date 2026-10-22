"""
Day 17: Local LLM backends — Ollama, and Apple's MLX as the alternative.

Both run on this machine and listen only on 127.0.0.1:
  - Ollama:  http://127.0.0.1:11434   (models like "qwen2.5:7b")
  - MLX:     http://127.0.0.1:8080    (models named "mlx:…"; start it with scripts/start_mlx_server.sh)

A model name starting with "mlx:" runs on MLX; anything else on Ollama. If the
requested Ollama model isn't available (Ollama not running, or it came up
without its models — e.g. after a restart before the external drive mounted),
resolve() falls back to the MLX model when the MLX server is up.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

OLLAMA_HOST = "http://127.0.0.1:11434"
MLX_HOST = "http://127.0.0.1:8080"
MLX_PREFIX = "mlx:"
# FinRAG name -> Hugging Face repo the MLX server was started with
MLX_MODELS = {"mlx:qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
              "mlx:qwen3-8b": "mlx-community/Qwen3-8B-4bit"}
# Written by scripts/start_mlx_server.sh: the one model the MLX server has loaded.
MLX_SERVING_PATH = Path(__file__).parent.parent / "data" / "mlx_serving.txt"
TIMEOUT_SEC = 600


def is_mlx(model: str) -> bool:
    return model.startswith(MLX_PREFIX)


def ollama_models() -> list[str]:
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        return []
    return [m["name"] for m in tags if not m.get("remote_host")]


def mlx_up() -> bool:
    try:
        return requests.get(f"{MLX_HOST}/health", timeout=3).ok
    except Exception:
        return False


def mlx_serving() -> str:
    """The MLX model the server was started with (default: the first one)."""
    try:
        name = MLX_SERVING_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        name = ""
    return name if name in MLX_MODELS else next(iter(MLX_MODELS))


def available_models() -> list[str]:
    """Ollama models (if Ollama is up) plus the MLX server's model (if it is up)."""
    return ollama_models() + ([mlx_serving()] if mlx_up() else [])


def resolve(model: str) -> str:
    """The model actually used: the requested one, or the MLX fallback if Ollama can't serve it."""
    if is_mlx(model):
        return model
    names = ollama_models()
    if model in names or f"{model}:latest" in names:
        return model
    if mlx_up():
        return mlx_serving()
    return model  # let the Ollama call report the problem


def _mlx_request(prompt: str, max_tokens: int, temperature: float, stream: bool, model: str):
    serving = mlx_serving()
    if model != serving:  # the server would load a second model on top — too much for 8 GB
        raise RuntimeError(f"The MLX server is serving {serving}, not {model}. Restart it with "
                           f"scripts/start_mlx_server.sh {model.removeprefix(MLX_PREFIX)}")
    return requests.post(
        f"{MLX_HOST}/v1/chat/completions",
        json={"model": MLX_MODELS.get(model, model), "stream": stream, "max_tokens": max_tokens,
              "temperature": temperature,
              "messages": [{"role": "user", "content": prompt}]},
        stream=stream, timeout=TIMEOUT_SEC,
    )


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Drop a reasoning block (Qwen3 "thinking" output) — a safety net; the server turns it off."""
    return THINK_RE.sub("", text)


def without_think(tokens):
    """Streaming version of strip_think: hold back a leading <think>…</think> block."""
    buf, done, trim = "", False, False  # trim: drop the blank lines that follow </think>
    for tok in tokens:
        if done:
            if trim:
                tok = tok.lstrip()
                trim = not tok
            if tok:
                yield tok
            continue
        buf += tok
        stripped = buf.lstrip()
        if not stripped:
            continue
        if not stripped.startswith("<think>") and not "<think>".startswith(stripped[:7]):
            done = True          # no reasoning block: release what we held
            yield buf
        elif "</think>" in buf:
            done = True
            rest = buf.split("</think>", 1)[1].lstrip()
            trim = not rest
            if rest:
                yield rest
    if not done and buf and "<think>" not in buf:
        yield buf


def mlx_stream(prompt: str, model: str, max_tokens: int, temperature: float = 0.0):
    yield from without_think(_mlx_stream(prompt, model, max_tokens, temperature))


def _mlx_stream(prompt: str, model: str, max_tokens: int, temperature: float = 0.0):
    """Yield answer tokens from the local MLX server (OpenAI-compatible streaming)."""
    resp = _mlx_request(prompt, max_tokens, temperature, True, model)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        delta = json.loads(data)["choices"][0].get("delta", {})
        if delta.get("content"):
            yield delta["content"]


def mlx_generate(prompt: str, model: str, max_tokens: int, temperature: float = 0.0) -> str:
    resp = _mlx_request(prompt, max_tokens, temperature, False, model)
    resp.raise_for_status()
    return strip_think(resp.json()["choices"][0]["message"]["content"])
