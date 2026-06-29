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

DEFAULT_SYSTEM_PROMPT = (
    "You are an anime image generation prompt expert. "
    "Convert the user's description into a concise comma-separated list of booru-style tags. "
    "CRITICAL FORMATTING RULES: "
    "Use spaces not underscores (write 'black hair' not 'black_hair'). "
    "Use only real standard booru tags - do not invent or guess tags. "
    "Focus ONLY on: character appearance (hair color, eye color, clothing, expression, pose, accessories), "
    "scene setting (location, time of day, weather, atmosphere, background elements), "
    "and specific art style details. "
    "Do NOT include quality tags like masterpiece, best quality, score_9, absurdres, etc. "
    "Those are handled separately. "
    "Output tags only - no explanation, no numbering, no markdown. /no_think"
)


class NeuralBooru:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "describe your scene here"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM_PROMPT
                }),
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": NOVA_ANIME_XL_TEMPLATE
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "qwen/qwen3-1.7b"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "max_tokens": ("INT", {
                    "default": 500, "min": 50, "max": 2000, "step": 50
                }),
                "lm_studio_url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234"
                }),
                "validate_tags": ("BOOLEAN", {"default": True}),
                "strict_tags": ("BOOLEAN", {"default": True}),
                "fuzzy_cutoff": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "min_post_count": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 100
                }),
                "max_tags": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "dropped_tags")
    FUNCTION = "generate"
    CATEGORY = "NeuralBooru"
    OUTPUT_NODE = False

    def generate(self, user_prompt, system_prompt, prompt_template, model,
                 temperature, max_tokens, lm_studio_url,
                 validate_tags=True, strict_tags=True, fuzzy_cutoff=0.0,
                 min_post_count=0, max_tags=0):

        url = lm_studio_url.rstrip("/") + "/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"].strip()

                # Strip <think>...</think> blocks (Qwen3 reasoning mode)
                if "<think>" in text:
                    text = re.sub(r"<think>.*?</think>", "", text,
                                  flags=re.DOTALL).strip()

                tags, dropped = self._validate(
                    text, validate_tags, strict_tags,
                    fuzzy_cutoff, min_post_count, max_tags)

                final = prompt_template.replace("{prompt}", tags)
                print(f"[NeuralBooru] {final[:160]}...")
                if dropped:
                    print(f"[NeuralBooru] dropped {len(dropped)} non-tags: {dropped}")
                return (final, ", ".join(dropped))

        except urllib.error.URLError as e:
            print(f"[NeuralBooru] Connection failed - is LM Studio running? ({e})")
            return (prompt_template.replace("{prompt}", user_prompt), "")
        except Exception as e:
            print(f"[NeuralBooru] Error: {e}")
            return (prompt_template.replace("{prompt}", user_prompt), "")

    def _validate(self, text, enabled, strict, fuzzy_cutoff,
                  min_post_count, max_tags):
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
        )
        return prompt, dropped


NODE_CLASS_MAPPINGS = {
    "NeuralBooru": NeuralBooru,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralBooru": "NeuralBooru",
}
