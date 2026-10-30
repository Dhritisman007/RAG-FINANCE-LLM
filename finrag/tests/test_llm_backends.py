"""
Backend-switch tests. No servers needed: the HTTP calls are faked.

    python tests/test_llm_backends.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import llm_backends as backends  # noqa: E402

API_ENV = ("PORTIQ_LLM_URL", "PORTIQ_LLM_MODEL", "PORTIQ_LLM_API_KEY", "PORTIQ_ALLOW_REMOTE_LLM")
for var in API_ENV:  # tests must not depend on (or use) a real configured server
    os.environ.pop(var, None)


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


def test_openai_compatible_api_backend() -> None:
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        body = {"choices": [{"message": {"content": "<think>\n</think>\n\nRevenue was ₹2,67,021 crore [1]."}}]}
        resp = FakeResponse(body)
        resp.status_code = 200
        return resp

    real = backends.requests.post
    try:
        backends.requests.post = post
        assert backends.api_config() is None                                       # nothing configured
        # A server on this machine (vLLM) works without any opt-in
        os.environ.update(PORTIQ_LLM_URL="http://127.0.0.1:8000/v1/", PORTIQ_LLM_MODEL="Qwen/Qwen3-8B")
        cfg = backends.api_config()
        assert cfg["name"] == "api:Qwen/Qwen3-8B" and cfg["local"] and cfg["allowed"]
        out = backends.api_generate("q", "api:Qwen/Qwen3-8B", 100)
        assert out == "Revenue was ₹2,67,021 crore [1]."                          # reasoning block stripped
        url, kw = sent[-1]
        assert url == "http://127.0.0.1:8000/v1/chat/completions" and "Authorization" not in kw["headers"]
        assert kw["json"]["model"] == "Qwen/Qwen3-8B" and kw["json"]["chat_template_kwargs"] == {"enable_thinking": False}

        # A remote server is refused unless explicitly allowed — nothing is sent
        os.environ.update(PORTIQ_LLM_URL="https://abc.endpoints.huggingface.cloud/v1", PORTIQ_LLM_API_KEY="hf_test")
        n = len(sent)
        try:
            backends.api_generate("q", "api:Qwen/Qwen3-8B", 100)
        except backends.RemoteLLMRefused as e:
            assert "abc.endpoints.huggingface.cloud" in str(e) and "PORTIQ_ALLOW_REMOTE_LLM=1" in str(e)
        else:
            raise AssertionError("a remote server must be refused without PORTIQ_ALLOW_REMOTE_LLM=1")
        assert len(sent) == n and not backends.api_config()["allowed"]            # nothing was sent

        # Allowed: bearer token sent; a non-Qwen3 model gets no chat-template arguments
        os.environ.update(PORTIQ_ALLOW_REMOTE_LLM="1", PORTIQ_LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct")
        backends.api_generate("q", "api:meta-llama/Llama-3.1-8B-Instruct", 100)
        url, kw = sent[-1]
        assert url == "https://abc.endpoints.huggingface.cloud/v1/chat/completions"
        assert kw["headers"]["Authorization"] == "Bearer hf_test" and "chat_template_kwargs" not in kw["json"]

        # Scaled-to-zero endpoint: a clear message instead of a raw 503
        def waking(url, **kwargs):
            resp = FakeResponse({})
            resp.status_code = 503
            return resp
        backends.requests.post = waking
        try:
            backends.api_generate("q", "api:meta-llama/Llama-3.1-8B-Instruct", 100)
        except RuntimeError as e:
            assert "starting up" in str(e)
        else:
            raise AssertionError("503 should say the server is starting")

        # Free-tier limits (Groq): 429 with a short Retry-After = busy for a minute; otherwise the daily quota
        for headers, daily, text in (({"retry-after": "7"}, False, "try again in 7 s"),
                                     ({}, True, "free daily limit is used up")):
            def limited(url, reply_headers=headers, **kwargs):
                resp = FakeResponse({})
                resp.status_code, resp.headers = 429, reply_headers
                return resp
            backends.requests.post = limited
            try:
                backends.api_generate("q", "api:meta-llama/Llama-3.1-8B-Instruct", 100)
            except backends.ProviderLimit as e:
                assert e.daily is daily and text in str(e), str(e)
            else:
                raise AssertionError("429 should raise ProviderLimit")
    finally:
        backends.requests.post = real
        for var in API_ENV:
            os.environ.pop(var, None)
    print("PASS: OpenAI-compatible server by URL; local allowed, remote refused unless PORTIQ_ALLOW_REMOTE_LLM=1; token, Qwen3 no-think, 503 message, 429 busy vs daily quota")


def test_api_fallback_order() -> None:
    real_get, real_path = backends.requests.get, backends.MLX_SERVING_PATH
    backends.MLX_SERVING_PATH = Path(tempfile.mkdtemp()) / "none.txt"
    try:
        os.environ.update(PORTIQ_LLM_URL="http://localhost:8000/v1", PORTIQ_LLM_MODEL="Qwen/Qwen3-8B")
        backends.requests.get = fake_get(None, mlx=False)            # deployment: no Ollama, no MLX
        assert backends.available_models() == ["api:Qwen/Qwen3-8B"]
        assert backends.resolve("qwen2.5:7b") == "api:Qwen/Qwen3-8B"
        backends.requests.get = fake_get(None, mlx=True)             # on the Mac, MLX is preferred
        assert backends.resolve("qwen2.5:7b") == "mlx:qwen2.5-7b"
        os.environ.update(PORTIQ_LLM_URL="https://example.com/v1")   # remote, not allowed: never offered
        backends.requests.get = fake_get(None, mlx=False)
        assert backends.available_models() == [] and backends.resolve("qwen2.5:7b") == "qwen2.5:7b"
    finally:
        backends.requests.get, backends.MLX_SERVING_PATH = real_get, real_path
        for var in API_ENV:
            os.environ.pop(var, None)
    print("PASS: fallback order Ollama → MLX → API; a disallowed remote API is never offered")


if __name__ == "__main__":
    test_resolve_and_fallback()
    test_mlx_stream_parses_sse()
    test_think_blocks_removed()
    test_openai_compatible_api_backend()
    test_api_fallback_order()
    print("\nAll backend tests passed.")
