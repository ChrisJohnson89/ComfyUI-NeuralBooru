"""
NeuralBooru - ComfyUI Custom Node
Converts natural language descriptions into booru-style anime prompt tags
using a local LLM via LM Studio's OpenAI-compatible API.
Default endpoint: http://localhost:1234/v1/chat/completions
"""

import json
import re
import urllib.request
import urllib.error

try:
    from . import tags as tagdb
except ImportError:  # standalone / direct import
    import tags as tagdb


NOVA_ANIME_XL_TEMPLATE = (
    "masterpiece, best quality, amazing quality, 4k, very aesthetic, "
    "high resolution, ultra-detailed, absurdres, newest, scenery, "
    "{prompt}, BREAK, depth of field, volumetric lighting"
)

# Quality-tag wrappers for common booru-trained model families. Selected via
# the template_preset input; "custom" uses the prompt_template text field.
TEMPLATE_PRESETS = {
    "nova_anime_xl": NOVA_ANIME_XL_TEMPLATE,
    "illustrious": ("masterpiece, best quality, very aesthetic, absurdres, "
                    "{prompt}"),
    "pony": "score_9, score_8_up, score_7_up, source_anime, {prompt}",
    "animagine_xl": ("{prompt}, masterpiece, best quality, very aesthetic, "
                     "absurdres"),
    "tags_only": "{prompt}",
}

DEFAULT_SYSTEM_PROMPT = (
    "You convert a scene description into Danbooru tags for an anime image model. "
    "Output ONLY lowercase tags separated by commas. "
    "No sentences, no explanations, no numbering, no category words. "
    "Write attributes in real Danbooru tag style: 'blue eyes' not 'eye color blue', "
    "'black hair' not 'hair color black'. Use spaces, not underscores. "
    "Use 1girl, 1boy, 2girls, etc. for people. "
    "Do NOT add quality tags like masterpiece, best quality, absurdres, score_9 - "
    "those are added separately.\n"
    "Example input: a cheerful blonde girl in a red dress on a beach at sunset\n"
    "Example output: 1girl, blonde hair, long hair, smile, red dress, beach, sunset, "
    "ocean, sky, cloud, standing\n"
    "Example input: a lone samurai in the rain at night\n"
    "Example output: 1boy, solo, samurai, japanese clothes, katana, rain, night, wet, "
    "serious, outdoors\n"
    "Now output tags only for the next description. /no_think"
)


class NeuralBooru:

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
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": NOVA_ANIME_XL_TEMPLATE,
                    "tooltip": "Final prompt wrapper; {prompt} is replaced with the "
                               "validated tags. If {prompt} is missing, tags are appended."
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
                # exists mainly so ComfyUI re-executes the node on demand;
                # also passed to the API (llama.cpp seeds are 32-bit)
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
                "validate_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check every tag against the real Danbooru vocabulary."
                }),
                "strict_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Drop candidates that match no real tag. Off keeps them raw."
                }),
                "fuzzy_cutoff": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Remap near-miss tags by similarity (0.8+ recommended). "
                               "0 disables fuzzy matching."
                }),
                "min_post_count": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 100,
                    "tooltip": "Drop tags with fewer Danbooru posts than this. 0 keeps all."
                }),
                "max_tags": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Keep at most this many tags. 0 means no limit."
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
                "exclude_categories": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Tag categories to drop, comma-separated: "
                               "artist, character, copyright, meta, general."
                }),
                "on_error": (["raise", "use_input_text"], {
                    "default": "raise",
                    "tooltip": "raise: fail the workflow with a visible error. "
                               "use_input_text: put your raw description in the template."
                }),
                "template_preset": (["custom"] + sorted(TEMPLATE_PRESETS), {
                    "default": "custom",
                    "tooltip": "Quality-tag wrapper for common model families. "
                               "custom uses the prompt_template field above."
                }),
                "sort_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reorder tags Danbooru-style: people counts, "
                               "character, copyright, artist, general, meta. "
                               "Booru-trained models respond to this order."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "dropped_tags")
    FUNCTION = "generate"
    CATEGORY = "NeuralBooru"
    OUTPUT_NODE = False

    def generate(self, user_prompt, system_prompt, prompt_template, model,
                 enable_thinking, temperature, max_tokens, seed, lm_studio_url,
                 validate_tags=True, strict_tags=True, fuzzy_cutoff=0.0,
                 min_post_count=0, max_tags=0, timeout=120, api_key="",
                 exclude_categories="", on_error="raise",
                 template_preset="custom", sort_tags=True):

        url = lm_studio_url.rstrip("/") + "/v1/chat/completions"
        prompt_template = TEMPLATE_PRESETS.get(template_preset, prompt_template)

        if enable_thinking:
            # let the switch win over the /no_think soft switch baked into
            # the default system prompt
            system_prompt = re.sub(r"\s*/no_think\s*$", "", system_prompt)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        if not enable_thinking:
            # honored by Ollama for Qwen3-style models; ignored elsewhere
            payload["enable_thinking"] = False

        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = "Bearer " + api_key.strip()

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                msg = result["choices"][0]["message"]
                text = (msg.get("content") or "").strip()
                if not text:
                    # Ollama (and DeepSeek-style servers) put reasoning-model
                    # output in a separate field, leaving content empty; the
                    # think-strip and validator below clean it up (issue #1)
                    text = (msg.get("reasoning") or
                            msg.get("reasoning_content") or "").strip()

                # Strip <think>...</think> blocks (Qwen3 reasoning mode).
                # An unterminated block (max_tokens cut off mid-reasoning) is
                # stripped to end of string so reasoning never leaks into tags.
                if "<think>" in text:
                    text = re.sub(r"<think>.*?(?:</think>|$)", "", text,
                                  flags=re.DOTALL).strip()

                tags, dropped = self._validate(
                    text, validate_tags, strict_tags,
                    fuzzy_cutoff, min_post_count, max_tags,
                    _parse_categories(exclude_categories), sort_tags)

                final = _apply_template(prompt_template, tags)
                print(f"[NeuralBooru] {final[:160]}...")
                if dropped:
                    print(f"[NeuralBooru] dropped {len(dropped)} non-tags: {dropped}")
                return (final, ", ".join(dropped))

        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            return self._fail(f"Server returned HTTP {e.code}: {detail}",
                              on_error, prompt_template, user_prompt, e)
        except urllib.error.URLError as e:
            return self._fail(
                f"Could not reach the LLM server at {lm_studio_url} ({e.reason}). "
                f"Is it running?", on_error, prompt_template, user_prompt, e)
        except Exception as e:
            return self._fail(f"Unexpected error: {e}",
                              on_error, prompt_template, user_prompt, e)

    def _fail(self, msg, on_error, prompt_template, user_prompt, exc):
        """Surface an LLM failure per on_error: raise it, or fall back to
        the raw input text so the workflow still produces an image."""
        if on_error == "raise":
            raise RuntimeError(f"NeuralBooru: {msg}") from exc
        print(f"[NeuralBooru] {msg} - falling back to raw input text")
        return (_apply_template(prompt_template, user_prompt), "")

    def _validate(self, text, enabled, strict, fuzzy_cutoff,
                  min_post_count, max_tags, exclude=(), sort_tags=False):
        """Filter LLM tags against the Danbooru vocabulary.

        Returns (tag_string, dropped_list). Falls back to the raw text if
        validation is off or the tag database could not be loaded.
        """
        if not enabled:
            return text, []
        db = tagdb.get_db()
        if db is None:
            return text, []
        prompt, _kept, dropped = db.validate(
            text,
            strict=strict,
            fuzzy_cutoff=fuzzy_cutoff,
            min_post_count=min_post_count,
            max_tags=max_tags,
            exclude_categories=exclude,
            sort_tags=sort_tags,
        )
        return prompt, dropped


def _apply_template(template, prompt):
    """Insert prompt into the template; append it if {prompt} is missing
    so tags are never silently discarded."""
    if "{prompt}" in template:
        return template.replace("{prompt}", prompt)
    print("[NeuralBooru] prompt_template has no {prompt} placeholder; appending")
    if not template.strip():
        return prompt
    return template.rstrip().rstrip(",") + ", " + prompt


def _parse_categories(spec):
    """Parse 'artist, meta' (or '1, 5') into a set of category ints.
    Unknown names are reported and skipped."""
    names = {v: k for k, v in tagdb.CATEGORY_NAMES.items()}
    out = set()
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part.isdigit():
            out.add(int(part))
        elif part in names:
            out.add(names[part])
        else:
            print(f"[NeuralBooru] unknown tag category {part!r} ignored "
                  f"(valid: {', '.join(names)})")
    return out


NODE_CLASS_MAPPINGS = {
    "NeuralBooru": NeuralBooru,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralBooru": "NeuralBooru",
}
