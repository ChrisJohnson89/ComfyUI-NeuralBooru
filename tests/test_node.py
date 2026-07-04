"""Tests for the NeuralBooru node (__init__.py) against a mocked server."""

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


def base_kwargs(node_module):
    return dict(
        user_prompt="a girl on a beach",
        system_prompt=node_module.DEFAULT_SYSTEM_PROMPT,
        prompt_template="{prompt}",
        model="test-model",
        enable_thinking=False,
        temperature=0.4,
        max_tokens=500,
        seed=42,
        lm_studio_url="http://localhost:1234",
    )


@pytest.fixture
def call(node_module, monkeypatch):
    """Run node.generate against a canned response; returns (output, request
    capture dict). Usage: out, req = call(response_body, **overrides).
    Pass ollama=True to emulate an Ollama server (probe answers, native API)."""
    def _call(body, exc=None, ollama=False, **overrides):
        captured = {}
        node_module.NeuralBooru._OLLAMA_CACHE.clear()

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/api/version"):  # the Ollama probe (GET)
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
        kwargs = base_kwargs(node_module)
        kwargs.update(overrides)
        out = node_module.NeuralBooru().generate(**kwargs)
        return out, captured

    return _call


CONTENT = {"choices": [{"message": {"content": "1girl, beach, smile"}}]}


# --- Ollama native path -------------------------------------------------------

def test_ollama_uses_native_api_with_think_control(call):
    (prompt, *_), req = call({"message": {"content": "1girl, beach, smile"}},
                             ollama=True)
    assert req["url"].endswith("/api/chat")
    p = req["payload"]
    assert p["think"] is False
    assert p["options"] == {"temperature": 0.4, "num_predict": 500, "seed": 42}
    assert prompt == "1girl, beach, smile"


def test_ollama_thinking_field_fallback(call):
    body = {"message": {"content": "", "thinking": "1girl, beach"}}
    (prompt, *_), _ = call(body, ollama=True)
    assert prompt == "1girl, beach"


def test_ollama_think_rejected_retries_without(node_module, monkeypatch):
    # models with no thinking mode 400 on the think flag; retry drops it
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

    node_module.NeuralBooru._OLLAMA_CACHE.clear()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    prompt, *_ = node_module.NeuralBooru().generate(**base_kwargs(node_module))
    assert prompt == "1girl, beach"
    assert len(payloads) == 2 and "think" not in payloads[1]


def test_non_ollama_uses_openai_endpoint(call):
    _, req = call(CONTENT)
    assert req["url"].endswith("/v1/chat/completions")


# --- request shape ----------------------------------------------------------

def test_payload_basics(call):
    (prompt, dropped, *_), req = call(CONTENT)
    p = req["payload"]
    assert p["model"] == "test-model"
    assert p["seed"] == 42
    assert p["enable_thinking"] is False
    assert p["messages"][0]["content"].endswith("/no_think")
    assert prompt == "1girl, beach, smile"
    assert dropped == ""


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
        "content": "<think>beach... girl...</think>1girl, beach"}}]}
    (prompt, *_), _ = call(body)
    assert prompt == "1girl, beach"


def test_unclosed_think_block_stripped(call):
    # regression: max_tokens truncation used to leak reasoning into the prompt
    body = {"choices": [{"message": {
        "content": "<think>okay the user wants a girl on a beach so"}}]}
    (prompt, *_), _ = call(body)
    assert "the user wants" not in prompt


def test_ollama_reasoning_fallback(call):
    # issue #1: Ollama leaves content empty, output lands in reasoning
    body = {"choices": [{"message": {
        "content": "", "reasoning": "1girl, beach, ocean"}}]}
    (prompt, *_), _ = call(body)
    assert prompt == "1girl, beach, ocean"


def test_null_content_reasoning_content_fallback(call):
    body = {"choices": [{"message": {
        "content": None, "reasoning_content": "1girl, night"}}]}
    (prompt, *_), _ = call(body)
    assert prompt == "1girl, night"


def test_validation_drops_junk_and_reports(call):
    body = {"choices": [{"message": {
        "content": "1girl, complete gibberish qzx, beach"}}]}
    (prompt, dropped, *_), _ = call(body)
    assert prompt == "1girl, beach"
    assert dropped == "complete gibberish qzx"


def test_exclude_categories_input(call):
    body = {"choices": [{"message": {"content": "1girl, highres"}}]}
    (prompt, dropped, *_), _ = call(body, exclude_categories="meta")
    assert prompt == "1girl"
    assert dropped == "highres"


def test_sort_tags_on_by_default(call):
    body = {"choices": [{"message": {"content": "night, hatsune miku, 1girl"}}]}
    (prompt, *_), _ = call(body)
    assert prompt == "1girl, hatsune miku, night"


def test_sort_tags_off_preserves_llm_order(call):
    body = {"choices": [{"message": {"content": "night, 1girl"}}]}
    (prompt, *_), _ = call(body, sort_tags=False)
    assert prompt == "night, 1girl"


def test_tags_output_is_untemplated(call):
    (prompt, dropped, tags), _ = call(
        CONTENT, prompt_template="quality, {prompt}, BREAK")
    assert prompt == "quality, 1girl, beach, smile, BREAK"
    assert tags == "1girl, beach, smile"


def test_tags_output_on_error_fallback(call):
    (_, _, tags), _ = call(None, exc=urllib.error.URLError("refused"),
                           on_error="use_input_text")
    assert tags == "a girl on a beach"


# --- template ----------------------------------------------------------------

def test_template_placeholder(call):
    (prompt, *_), _ = call(CONTENT, prompt_template="quality, {prompt}, BREAK")
    assert prompt == "quality, 1girl, beach, smile, BREAK"


def test_template_without_placeholder_appends(call):
    (prompt, *_), _ = call(CONTENT, prompt_template="masterpiece, best quality")
    assert prompt == "masterpiece, best quality, 1girl, beach, smile"


def test_template_preset_overrides_field(call):
    (prompt, *_), _ = call(CONTENT, template_preset="pony",
                          prompt_template="ignored {prompt}")
    assert prompt.startswith("score_9, score_8_up")
    assert "ignored" not in prompt


def test_template_preset_custom_uses_field(call):
    (prompt, *_), _ = call(CONTENT, template_preset="custom",
                          prompt_template="X, {prompt}")
    assert prompt == "X, 1girl, beach, smile"


def test_all_presets_have_placeholder(node_module):
    for name, template in node_module.TEMPLATE_PRESETS.items():
        assert "{prompt}" in template, name


def test_parse_categories(node_module):
    assert node_module._parse_categories("artist, meta, 4") == {1, 5, 4}
    assert node_module._parse_categories("") == set()
    assert node_module._parse_categories("bogus") == set()


# --- error handling ------------------------------------------------------------

def test_connection_error_raises_by_default(call):
    with pytest.raises(RuntimeError, match="Could not reach"):
        call(None, exc=urllib.error.URLError("connection refused"))


def test_http_error_includes_status_and_body(call):
    err = urllib.error.HTTPError(
        "http://x", 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))
    with pytest.raises(RuntimeError, match=r"HTTP 401.*bad key"):
        call(None, exc=err)


def test_on_error_use_input_text_falls_back(call):
    (prompt, dropped, *_), _ = call(
        None, exc=urllib.error.URLError("refused"),
        on_error="use_input_text",
        prompt_template="quality, {prompt}")
    assert prompt == "quality, a girl on a beach"
    assert dropped == ""
