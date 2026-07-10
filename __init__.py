"""
NeuralBooru - ComfyUI Custom Node
Converts natural language descriptions into booru-style anime prompt tags
using a local LLM via LM Studio's OpenAI-compatible API.
Default endpoint: http://localhost:1234/v1/chat/completions
"""

import json
import urllib.request

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


# base URL -> bool, probed once per server per session. Shared with
# llm_node so the combined and split nodes never probe twice.
_OLLAMA_CACHE = {}


def _is_ollama(base):
    """Detect an Ollama server by its native /api/version endpoint.
    LM Studio and other OpenAI-compatible servers 404 this instantly."""
    cached = _OLLAMA_CACHE.get(base)
    if cached is not None:
        return cached
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=5):
            found = True
    except Exception:
        found = False
    _OLLAMA_CACHE[base] = found
    return found


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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "dropped_tags", "tags")
    OUTPUT_TOOLTIPS = (
        "Validated tags wrapped in the prompt template, for your CLIP encoder.",
        "Candidates that failed validation, comma-separated.",
        "The validated tags alone, without the template.",
    )
    FUNCTION = "generate"
    CATEGORY = "NeuralBooru"
    OUTPUT_NODE = False

    # legacy alias: tests and old code reach the probe cache through the class
    _OLLAMA_CACHE = _OLLAMA_CACHE

    def generate(self, user_prompt, system_prompt, prompt_template, model,
                 enable_thinking, temperature, max_tokens, seed, lm_studio_url,
                 validate_tags=True, strict_tags=True, fuzzy_cutoff=0.0,
                 min_post_count=0, max_tags=0, timeout=120, api_key="",
                 exclude_categories="", on_error="raise",
                 template_preset="custom", sort_tags=True):
        # Thin wrapper over the split nodes; the LLM and validation logic
        # live only in llm_node.py / validator_node.py.
        try:
            raw_tags, is_fallback = NeuralBooruLLM().generate(
                user_prompt, system_prompt, model, enable_thinking,
                temperature, max_tokens, seed, lm_studio_url,
                timeout=timeout, api_key=api_key, on_error=on_error)
        except RuntimeError as e:
            raise RuntimeError(
                str(e).replace("NeuralBooruLLM:", "NeuralBooru:", 1)) from e
        return NeuralBooruValidator().generate(
            raw_tags, prompt_template, is_fallback=is_fallback,
            validate_tags=validate_tags, strict_tags=strict_tags,
            fuzzy_cutoff=fuzzy_cutoff, min_post_count=min_post_count,
            max_tags=max_tags, exclude_categories=exclude_categories,
            template_preset=template_preset, sort_tags=sort_tags)


def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_error_detail(e):
    """Best-effort response body from an HTTPError (readable only once)."""
    try:
        return e.read().decode("utf-8", "replace")[:300]
    except Exception:
        return ""


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

# NeuralBooru.generate delegates to these, so this import is load-bearing.
# The absolute fallback covers standalone loading under the name
# "neuralbooru" (tests). The finetune tooling loads __init__.py under a
# different module name purely for its constants; there the split modules
# cannot resolve their imports back to this module, so they stay None -
# fine as long as generate() is never called in that context.
try:
    from .llm_node import NeuralBooruLLM
    from .validator_node import NeuralBooruValidator
except ImportError:
    try:
        from llm_node import NeuralBooruLLM
        from validator_node import NeuralBooruValidator
    except ImportError:
        NeuralBooruLLM = NeuralBooruValidator = None

if NeuralBooruLLM is not None:
    NODE_CLASS_MAPPINGS["NeuralBooruLLM"] = NeuralBooruLLM
    NODE_CLASS_MAPPINGS["NeuralBooruValidator"] = NeuralBooruValidator
    NODE_DISPLAY_NAME_MAPPINGS["NeuralBooruLLM"] = "NeuralBooru LLM"
    NODE_DISPLAY_NAME_MAPPINGS["NeuralBooruValidator"] = "NeuralBooru Validator"
