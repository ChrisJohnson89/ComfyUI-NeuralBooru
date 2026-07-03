"""
Shared data-factory plumbing for NeuralBooru finetuning (Phase 1).

This mirrors the live node (../__init__.py) exactly so that training data is
generated under the same conditions the model will see at inference:
  - same DEFAULT_SYSTEM_PROMPT (single source of truth, loaded from the node)
  - same chat payload shape against LM Studio's OpenAI-compatible API
  - same <think>...</think> stripping
  - same validation via ../tags.py (get_db().validate)

Nothing here is imported by the ComfyUI node; it is offline tooling.
"""

import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tags as tagdb  # noqa: E402  (resolved via ROOT on sys.path)

# Also matches an unterminated block (output truncated mid-reasoning) so
# reasoning text never leaks into training data.
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def _load_node():
    """Load ../__init__.py by path to reuse DEFAULT_SYSTEM_PROMPT verbatim.

    Loading by path (rather than re-declaring the prompt here) keeps a single
    source of truth: if the node's system prompt changes, the training data
    follows automatically.
    """
    spec = importlib.util.spec_from_file_location(
        "neuralbooru_node", os.path.join(ROOT, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_NODE = _load_node()
DEFAULT_SYSTEM_PROMPT = _NODE.DEFAULT_SYSTEM_PROMPT


class LMStudio:
    """Minimal client for LM Studio's OpenAI-compatible endpoints."""

    def __init__(self, base_url="http://localhost:1234", timeout=120):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def models(self):
        """Return the list of currently loaded model ids, or [] on failure."""
        try:
            with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def chat(self, model, system, user, temperature=0.4, max_tokens=500):
        """One chat completion. Returns assistant text with <think> stripped."""
        result = self._post("/v1/chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        })
        msg = result["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text:
            # Ollama-style servers put reasoning-model output in a separate
            # field, leaving content empty (node issue #1)
            text = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        if "<think>" in text:
            text = _THINK.sub("", text).strip()
        return text


def validate(text, strict=True, fuzzy_cutoff=0.0, min_post_count=0, max_tags=0):
    """Run raw LLM output through the node's tag validator.

    Returns (clean_string, kept_list, dropped_list). If the tag DB is
    unavailable this raises, because for dataset building we never want to
    silently keep unvalidated text.
    """
    db = tagdb.get_db()
    if db is None:
        raise RuntimeError("Danbooru tag DB unavailable; cannot validate")
    prompt, kept, dropped = db.validate(
        text,
        strict=strict,
        fuzzy_cutoff=fuzzy_cutoff,
        min_post_count=min_post_count,
        max_tags=max_tags,
    )
    return prompt, kept, dropped
