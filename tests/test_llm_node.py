"""Tests for the NeuralBooruLLM node (llm_node.py) against a mocked server."""

import io
import json
import urllib.error
import urllib.request

import pytest


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def base_kwargs(llm_module):
    return dict(
        user_prompt="a girl on a beach",
        system_prompt=llm_module.DEFAULT_SYSTEM_PROMPT,
        model="test-model",
        enable_thinking=False,
        temperature=0.4,
        max_tokens=500,
        seed=42,
        lm_studio_url="http://localhost:1234",
    )


@pytest.fixture
def call(llm_module, monkeypatch):
    """Run NeuralBooruLLM.generate against a canned response; returns
    ((raw_tags, is_fallback), request_capture). Pass ollama=True to emulate
    an Ollama server (probe answers, native API)."""
    def _call(body, exc=None, ollama=False, **overrides):
        captured = {}
        llm_module._OLLAMA_CACHE.clear()

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/api/version"):
                if ollama:
                    return FakeResponse({"version": "0.0.0-test"})
                raise urllib.error.HTTPError(url, 404, "Not Found", {},
                                             io.BytesIO(b""))
            captured["url"] = url
            captured["timeout"] = timeout
            captured["headers"] = dict(req.headers)
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            if exc is not None:
                raise exc
            return FakeResponse(body)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        kwargs = base_kwargs(llm_module)
        kwargs.update(overrides)
        out = llm_module.NeuralBooruLLM().generate(**kwargs)
        return out, captured

    return _call


CONTENT = {"choices": [{"message": {"content": "1girl, beach, smile"}}]}


# --- Ollama native path -------------------------------------------------------

def test_ollama_uses_native_api_with_think_control(call):
    (raw_tags, is_fallback), req = call(
        {"message": {"content": "1girl, beach, smile"}}, ollama=True)
    assert req["url"].endswith("/api/chat")
    p = req["payload"]
    assert p["think"] is False
    assert p["options"] == {"temperature": 0.4, "num_predict": 500, "seed": 42}
    assert raw_tags == "1girl, beach, smile"
    assert is_fallback is False


def test_ollama_thinking_field_fallback(call):
    body = {"message": {"content": "", "thinking": "1girl, beach"}}
    (raw_tags, _), _ = call(body, ollama=True)
    assert raw_tags == "1girl, beach"


def test_ollama_think_rejected_retries_without(llm_module, monkeypatch):
    payloads = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return FakeResponse({"version": "0.0.0-test"})
        p = json.loads(req.data.decode("utf-8"))
        payloads.append(p)
        if "think" in p:
            raise urllib.error.HTTPError(
                url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":"\\"test-model\\" does not support thinking"}'))
        return FakeResponse({"message": {"content": "1girl, beach"}})

    llm_module._OLLAMA_CACHE.clear()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    raw_tags, is_fallback = llm_module.NeuralBooruLLM().generate(
        **base_kwargs(llm_module))
    assert raw_tags == "1girl, beach"
    assert is_fallback is False
    assert len(payloads) == 2 and "think" not in payloads[1]


def test_non_ollama_uses_openai_endpoint(call):
    _, req = call(CONTENT)
    assert req["url"].endswith("/v1/chat/completions")


# --- request shape ----------------------------------------------------------

def test_payload_basics(call):
    (raw_tags, _), req = call(CONTENT)
    p = req["payload"]
    assert p["model"] == "test-model"
    assert p["seed"] == 42
    assert p["enable_thinking"] is False
    assert p["messages"][0]["content"].endswith("/no_think")
    assert raw_tags == "1girl, beach, smile"


def test_enable_thinking_on_strips_no_think(call):
    _, req = call(CONTENT, enable_thinking=True)
    p = req["payload"]
    assert "enable_thinking" not in p
    assert "/no_think" not in p["messages"][0]["content"]


def test_timeout_and_api_key(call):
    _, req = call(CONTENT, timeout=45, api_key="sk-test")
    assert req["timeout"] == 45
    assert req["headers"].get("Authorization") == "Bearer sk-test"


def test_no_auth_header_when_api_key_blank(call):
    _, req = call(CONTENT, api_key="  ")
    assert "Authorization" not in req["headers"]


# --- response handling --------------------------------------------------------

def test_think_block_stripped(call):
    body = {"choices": [{"message": {
        "content": "<think>okay beach... girl...</think>1girl, beach"}}]}
    (raw_tags, _), _ = call(body)
    assert raw_tags == "1girl, beach"


def test_unclosed_think_block_stripped(call):
    body = {"choices": [{"message": {
        "content": "<think>okay the user wants a girl on a beach so"}}]}
    (raw_tags, _), _ = call(body)
    assert "the user wants" not in raw_tags


def test_ollama_reasoning_fallback(call):
    body = {"choices": [{"message": {
        "content": "", "reasoning": "1girl, beach, ocean"}}]}
    (raw_tags, _), _ = call(body)
    assert raw_tags == "1girl, beach, ocean"


def test_null_content_reasoning_content_fallback(call):
    body = {"choices": [{"message": {
        "content": None, "reasoning_content": "1girl, night"}}]}
    (raw_tags, _), _ = call(body)
    assert raw_tags == "1girl, night"


# --- error handling ----------------------------------------------------------

def test_connection_error_raises_by_default(call):
    with pytest.raises(RuntimeError, match="Could not reach"):
        call(None, exc=urllib.error.URLError("connection refused"))


def test_http_error_includes_status_and_body(call):
    err = urllib.error.HTTPError(
        "http://x", 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))
    with pytest.raises(RuntimeError, match=r"HTTP 401.*bad key"):
        call(None, exc=err)


def test_on_error_use_input_text_falls_back(call):
    (raw_tags, is_fallback), _ = call(
        None, exc=urllib.error.URLError("refused"),
        on_error="use_input_text")
    assert raw_tags == "a girl on a beach"
    assert is_fallback is True
