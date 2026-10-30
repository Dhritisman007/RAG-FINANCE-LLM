"""
LLM backends — Ollama and Apple's MLX on this Mac (Day 17), and any
OpenAI-compatible server by URL (Day 19: vLLM, a Hugging Face Inference
Endpoint, …) for hosting.

  - Ollama:  http://127.0.0.1:11434   (models like "qwen2.5:7b")
  - MLX:     http://127.0.0.1:8080    (models named "mlx:…"; start it with scripts/start_mlx_server.sh)
  - API:     PORTIQ_LLM_URL           (the model named "api:<PORTIQ_LLM_MODEL>")

API settings (environment variables):
  PORTIQ_LLM_URL           OpenAI-style base URL, e.g. http://127.0.0.1:8000/v1 (vLLM on this
                           machine) or https://<name>.endpoints.huggingface.cloud/v1
  PORTIQ_LLM_MODEL         the model id the server expects, e.g. Qwen/Qwen3-8B
  PORTIQ_LLM_API_KEY       optional bearer token (a Hugging Face token for HF endpoints)
  PORTIQ_ALLOW_REMOTE_LLM  must be "1" for a server that isn't on this machine: prompts —
                           questions, retrieved passages and holding details — are then sent
                           to it. Without it a remote URL is refused, so configuring a URL
                           alone can never send data off the machine.

If the requested Ollama model isn't available (Ollama not running, or it came up
without its models — e.g. after a restart before the external drive mounted),
resolve() falls back to the MLX model when the MLX server is up, else to the API model.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

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
API_PREFIX = "api:"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ProviderLimit(RuntimeError):
    """The model provider refused for rate or quota (HTTP 429). `daily` = come back tomorrow, else in a minute."""

    def __init__(self, message: str, daily: bool):
        super().__init__(message)
        self.daily = daily


class RemoteLLMRefused(RuntimeError):
    """A remote model server is configured but sending prompts off this machine wasn't allowed."""


def is_mlx(model: str) -> bool:
    return model.startswith(MLX_PREFIX)


def is_api(model: str) -> bool:
    return model.startswith(API_PREFIX)


def api_config() -> dict | None:
    """The OpenAI-compatible server from the environment, or None if not configured."""
    url = os.environ.get("PORTIQ_LLM_URL", "").strip().rstrip("/")
    model = os.environ.get("PORTIQ_LLM_MODEL", "").strip()
    if not url or not model:
        return None
    host = urlparse(url).hostname or ""
    return {"url": url, "model": model, "name": API_PREFIX + model, "host": host,
            "key": os.environ.get("PORTIQ_LLM_API_KEY", "").strip(),
            "local": host in LOCAL_HOSTS,
            "allowed": host in LOCAL_HOSTS or os.environ.get("PORTIQ_ALLOW_REMOTE_LLM", "").strip() == "1"}


def check_api_allowed(cfg: dict) -> None:
    if not cfg["allowed"]:
        raise RemoteLLMRefused(
            f"PORTIQ_LLM_URL points to {cfg['host']}, which is not this machine. Questions, retrieved passages "
            f"and holding details would be sent there. Set PORTIQ_ALLOW_REMOTE_LLM=1 only if that is intended.")


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
    """Ollama models (if up), the MLX server's model (if up) and the API model (if configured and allowed).
    The API server isn't probed: a scaled-to-zero endpoint would look "down" until its first request."""
    cfg = api_config()
    return (ollama_models() + ([mlx_serving()] if mlx_up() else [])
            + ([cfg["name"]] if cfg and cfg["allowed"] else []))


def resolve(model: str) -> str:
    """The model actually used: the requested one, or the MLX fallback if Ollama can't serve it."""
    if is_mlx(model) or is_api(model):
        return model
    names = ollama_models()
    if model in names or f"{model}:latest" in names:
        return model
    if mlx_up():
        return mlx_serving()
    cfg = api_config()
    if cfg and cfg["allowed"]:
        return cfg["name"]
    return model  # let the Ollama call report the problem


def _chat_request(base_url: str, model_id: str, prompt: str, max_tokens: int, temperature: float,
                  stream: bool, headers: dict | None = None, extra: dict | None = None):
    """POST an OpenAI-style chat completion (MLX and API servers both speak it)."""
    body = {"model": model_id, "stream": stream, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}], **(extra or {})}
    return requests.post(f"{base_url}/chat/completions", json=body, headers=headers or {},
                         stream=stream, timeout=TIMEOUT_SEC)


def _sse_tokens(resp):
    """Content tokens from an OpenAI-style streaming (server-sent events) response."""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        choices = json.loads(data).get("choices") or [{}]
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            yield delta["content"]


def _mlx_request(prompt: str, max_tokens: int, temperature: float, stream: bool, model: str):
    serving = mlx_serving()
    if model != serving:  # the server would load a second model on top — too much for 8 GB
        raise RuntimeError(f"The MLX server is serving {serving}, not {model}. Restart it with "
                           f"scripts/start_mlx_server.sh {model.removeprefix(MLX_PREFIX)}")
    return _chat_request(f"{MLX_HOST}/v1", MLX_MODELS.get(model, model), prompt, max_tokens, temperature, stream)


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
    yield from _sse_tokens(resp)


def mlx_generate(prompt: str, model: str, max_tokens: int, temperature: float = 0.0) -> str:
    resp = _mlx_request(prompt, max_tokens, temperature, False, model)
    resp.raise_for_status()
    return strip_think(resp.json()["choices"][0]["message"]["content"])


# ── Any OpenAI-compatible server (vLLM, Hugging Face Inference Endpoints, …) ──
def _api_request(prompt: str, max_tokens: int, temperature: float, stream: bool, model: str):
    cfg = api_config()
    if not cfg or model != cfg["name"]:
        raise RuntimeError(f"{model} isn't the configured API model — set PORTIQ_LLM_URL and PORTIQ_LLM_MODEL.")
    check_api_allowed(cfg)
    headers = {"Authorization": f"Bearer {cfg['key']}"} if cfg["key"] else {}
    # Qwen3 "thinks" before answering unless told not to; vLLM passes this to the chat template.
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in cfg["model"].lower() else None
    resp = _chat_request(cfg["url"], cfg["model"], prompt, max_tokens, temperature, stream, headers, extra)
    if resp.status_code == 503:  # e.g. a Hugging Face endpoint scaled to zero, still starting
        raise RuntimeError(f"The model server at {cfg['host']} is starting up — try again in a minute or two.")
    if resp.status_code == 429:  # free-tier limits (Groq: per minute and per day, per organisation)
        # Retry-After (seconds) tells a short per-minute limit from a daily one. TODO, unverified: whether
        # Groq always sends it on 429 — without it, assume the daily quota (the safer message).
        try:
            wait = float(resp.headers.get("retry-after", ""))
        except ValueError:
            wait = None
        if wait is not None and wait <= 120:
            raise ProviderLimit(f"The AI service is busy — try again in {max(1, round(wait))} s.", daily=False)
        raise ProviderLimit("The AI service's free daily limit is used up — try again tomorrow.", daily=True)
    resp.raise_for_status()
    return resp


def api_stream(prompt: str, model: str, max_tokens: int, temperature: float = 0.0):
    """Yield answer tokens from the configured OpenAI-compatible server."""
    yield from without_think(_sse_tokens(_api_request(prompt, max_tokens, temperature, True, model)))


def api_generate(prompt: str, model: str, max_tokens: int, temperature: float = 0.0) -> str:
    resp = _api_request(prompt, max_tokens, temperature, False, model)
    return strip_think(resp.json()["choices"][0]["message"]["content"])
