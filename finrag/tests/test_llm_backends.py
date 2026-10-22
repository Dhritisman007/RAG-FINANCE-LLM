"""
Backend-switch tests. No servers needed: the HTTP calls are faked.

    python tests/test_llm_backends.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import llm_backends as backends  # noqa: E402


class FakeResponse:
    def __init__(self, payload=None, ok=True, lines=()):
        self.payload, self.ok, self.lines = payload, ok, lines

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)


def fake_get(ollama, mlx):
    """ollama: list of model names, or None if Ollama is down; mlx: whether the MLX server is up."""
    def get(url, **kwargs):
        assert url.startswith("http://127.0.0.1:"), url  # only ever local
        if url.startswith(backends.OLLAMA_HOST):
            if ollama is None:
                raise ConnectionError
            return FakeResponse({"models": [{"name": n} for n in ollama]})
        if not mlx:
            raise ConnectionError
        return FakeResponse({"data": []})
    return get


def test_resolve_and_fallback() -> None:
    real = backends.requests.get
    try:
        backends.requests.get = fake_get(["qwen2.5:7b"], mlx=True)
        assert backends.resolve("qwen2.5:7b") == "qwen2.5:7b"
        assert backends.available_models() == ["qwen2.5:7b", "mlx:qwen2.5-7b"]
        backends.requests.get = fake_get([], mlx=True)          # Ollama up with no models
        assert backends.resolve("qwen2.5:7b") == "mlx:qwen2.5-7b"
        backends.requests.get = fake_get(None, mlx=True)        # Ollama not running
        assert backends.resolve("qwen2.5:7b") == "mlx:qwen2.5-7b"
        assert backends.available_models() == ["mlx:qwen2.5-7b"]
        backends.requests.get = fake_get(None, mlx=False)       # nothing up: leave it to Ollama's error
        assert backends.resolve("qwen2.5:7b") == "qwen2.5:7b" and backends.available_models() == []
    finally:
        backends.requests.get = real
    print("PASS: Ollama models used when present; MLX fallback when Ollama is down or empty")


def test_mlx_stream_parses_sse() -> None:
    sent = []
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
              {"choices": [{"delta": {"content": "Hello"}}]},
              {"choices": [{"delta": {"content": " world"}}]}]
    lines = [""] + [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return FakeResponse(lines=lines)

    real = backends.requests.post
    try:
        backends.requests.post = post
        assert "".join(backends.mlx_stream("q", "mlx:qwen2.5-7b", 50)) == "Hello world"
    finally:
        backends.requests.post = real
    url, kwargs = sent[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "mlx-community/Qwen2.5-7B-Instruct-4bit" and body["max_tokens"] == 50
    assert body["temperature"] == 0.0 and body["stream"] is True
    print("PASS: MLX streaming request goes to 127.0.0.1 and its tokens are parsed")


if __name__ == "__main__":
    test_resolve_and_fallback()
    test_mlx_stream_parses_sse()
    print("\nAll backend tests passed.")
