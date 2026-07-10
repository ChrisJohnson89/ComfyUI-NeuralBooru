"""
NeuralBooruLLM - LLM prompt transformation node.

Calls an OpenAI-compatible LLM server (LM Studio, Ollama, llama.cpp, vLLM)
and returns raw tag-ish text. Does no Danbooru validation or template
wrapping; pair with NeuralBooruValidator for the full original pipeline.

When wired in sequence:  NeuralBooruLLM.raw_tags -> NeuralBooruValidator.tags
the two nodes reproduce the original NeuralBooru node exactly.
"""

import re
import urllib.error

try:
    from . import (DEFAULT_SYSTEM_PROMPT, _post_json, _http_error_detail,
                   _is_ollama, _OLLAMA_CACHE)
except ImportError:
    from neuralbooru import (DEFAULT_SYSTEM_PROMPT, _post_json,
                             _http_error_detail, _is_ollama, _OLLAMA_CACHE)


class NeuralBooruLLM:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "describe your scene here",
                    "tooltip": "Plain-English scene description to convert into tags."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM_PROMPT,
                    "tooltip": "Instructions for the LLM. Edit to change tagging behavior."
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "qwen/qwen3-1.7b",
                    "tooltip": "Model id as the server reports it (see /v1/models)."
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow reasoning-mode models to think before answering. "
                               "Slower; output is cleaned either way."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "max_tokens": ("INT", {
                    "default": 500, "min": 50, "max": 2000, "step": 50
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffff,
                    "control_after_generate": True,
                    "tooltip": "Change to re-roll the LLM call; also sent to the API."
                }),
                "lm_studio_url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234",
                    "tooltip": "Base URL of any OpenAI-compatible server: "
                               "LM Studio (:1234), Ollama (:11434), llama.cpp, vLLM."
                }),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 120, "min": 10, "max": 600, "step": 10,
                    "tooltip": "Seconds to wait for the LLM response."
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Optional Bearer token for servers that require auth "
                               "(vLLM, remote endpoints). Leave empty for LM Studio/Ollama."
                }),
                "on_error": (["raise", "use_input_text"], {
                    "default": "raise",
                    "tooltip": "raise: fail the workflow with a visible error. "
                               "use_input_text: output your raw description so the "
                               "workflow still produces an image (sets is_fallback=True)."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("raw_tags", "is_fallback")
    OUTPUT_TOOLTIPS = (
        "The LLM's tag-ish output, think-stripped and reasoning-fallback applied. "
        "Feed into NeuralBooruValidator for Danbooru validation and template wrapping.",
        "True when the LLM call failed and raw_tags holds the original user_prompt "
        "(on_error=use_input_text). NeuralBooruValidator skips validation in this case.",
    )
    FUNCTION = "generate"
    CATEGORY = "NeuralBooru"
    OUTPUT_NODE = False

    def generate(self, user_prompt, system_prompt, model,
                 enable_thinking, temperature, max_tokens, seed, lm_studio_url,
                 timeout=120, api_key="", on_error="raise"):

        base = lm_studio_url.rstrip("/")

        if enable_thinking:
            system_prompt = re.sub(r"\s*/no_think\s*$", "", system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = "Bearer " + api_key.strip()

        if _is_ollama(base):
            url = base + "/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "think": bool(enable_thinking),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "seed": seed,
                },
            }
        else:
            url = base + "/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "stream": False,
            }
            if not enable_thinking:
                payload["enable_thinking"] = False

        try:
            try:
                result = _post_json(url, payload, headers, timeout)
            except urllib.error.HTTPError as e:
                if ("think" in payload and e.code == 400
                        and "think" in _http_error_detail(e).lower()):
                    payload.pop("think")
                    result = _post_json(url, payload, headers, timeout)
                else:
                    raise

            msg = (result["message"] if "message" in result
                   else result["choices"][0]["message"])
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("thinking") or msg.get("reasoning") or
                        msg.get("reasoning_content") or "").strip()

            # Strip think blocks (Qwen3 reasoning mode).
            # An unterminated block (max_tokens cut off mid-reasoning) is
            # stripped to end of string so reasoning never leaks into tags.
            if "<think>" in text:
                text = re.sub(r"<think>.*?(?:</think>|$)", "", text,
                              flags=re.DOTALL).strip()

            print(f"[NeuralBooruLLM] {text[:160]}...")
            return (text, False)

        except urllib.error.HTTPError as e:
            return _fail(
                f"Server returned HTTP {e.code}: {_http_error_detail(e)}",
                on_error, user_prompt, e)
        except urllib.error.URLError as e:
            return _fail(
                f"Could not reach the LLM server at {lm_studio_url} ({e.reason}). "
                f"Is it running?", on_error, user_prompt, e)
        except Exception as e:
            return _fail(f"Unexpected error: {e}", on_error, user_prompt, e)


def _fail(msg, on_error, user_prompt, exc):
    """Surface an LLM failure per on_error: raise it, or fall back to
    the raw input text (is_fallback=True) so the workflow still produces
    an image."""
    if on_error == "raise":
        raise RuntimeError(f"NeuralBooruLLM: {msg}") from exc
    print(f"[NeuralBooruLLM] {msg} - falling back to raw input text")
    return (user_prompt, True)
