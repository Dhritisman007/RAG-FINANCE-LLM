"""
Backend-switch tests. No servers needed: the HTTP calls are faked.

    python tests/test_llm_backends.py
"""
import json
import sys
import tempfile
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
    real, real_path = backends.requests.get, backends.MLX_SERVING_PATH
    backends.MLX_SERVING_PATH = Path(tempfile.mkdtemp()) / "none.txt"   # no state file: the default model
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
        backends.MLX_SERVING_PATH.write_text("mlx:qwen3-8b\n")       # the server was started with Qwen3
        backends.requests.get = fake_get(None, mlx=True)
        assert backends.available_models() == ["mlx:qwen3-8b"] and backends.resolve("qwen2.5:7b") == "mlx:qwen3-8b"
    finally:
        backends.requests.get, backends.MLX_SERVING_PATH = real, real_path
    print("PASS: Ollama models used when present; MLX fallback when down or empty; only the served MLX model is offered")


def test_mlx_stream_parses_sse() -> None:
    sent = []
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
              {"choices": [{"delta": {"content": "Hello"}}]},
              {"choices": [{"delta": {"content": " world"}}]}]
    lines = [""] + [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return FakeResponse(lines=lines)

    real, real_path = backends.requests.post, backends.MLX_SERVING_PATH
    backends.MLX_SERVING_PATH = Path(tempfile.mkdtemp()) / "none.txt"
    try:
        backends.requests.post = post
        assert "".join(backends.mlx_stream("q", "mlx:qwen2.5-7b", 50)) == "Hello world"
        try:
            backends.mlx_generate("q", "mlx:qwen3-8b", 10)   # not the served model (default qwen2.5-7b)
        except RuntimeError as e:
            assert "start_mlx_server.sh qwen3-8b" in str(e)
        else:
            raise AssertionError("asking for a model the server isn't serving must fail, not load a second model")
    finally:
        backends.requests.post, backends.MLX_SERVING_PATH = real, real_path
    assert len(sent) == 1                                 # the refused request never reached the server
    url, kwargs = sent[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "mlx-community/Qwen2.5-7B-Instruct-4bit" and body["max_tokens"] == 50
    assert body["temperature"] == 0.0 and body["stream"] is True
    print("PASS: MLX streaming request goes to 127.0.0.1 and its tokens are parsed; only the served model is used")


def test_think_blocks_removed() -> None:
    assert backends.strip_think("<think>\nlet me see\n</think>\n\nRevenue was ₹1 [1].") == "Revenue was ₹1 [1]."
    assert backends.strip_think("No reasoning here [2].") == "No reasoning here [2]."
    cases = [
        (["<th", "ink>", "plan", "</th", "ink>", "\n\nAnswer", " [1]."], "Answer [1]."),   # split across tokens
        (["\n", "Answer", " is ", "here."], "\nAnswer is here."),                          # no block: unchanged
        (["<table>", " rows"], "<table> rows"),                                           # '<' that isn't <think>
        (["<think>", "never closed"], ""),                                                # unfinished reasoning dropped
    ]
    for tokens, want in cases:
        got = "".join(backends.without_think(iter(tokens)))
        assert got == want, (tokens, got)
    print("PASS: reasoning blocks are stripped from streamed and complete answers; other text untouched")


if __name__ == "__main__":
    test_resolve_and_fallback()
    test_mlx_stream_parses_sse()
    test_think_blocks_removed()
    print("\nAll backend tests passed.")
