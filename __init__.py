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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "NeuralBooru"
    OUTPUT_NODE = False

    def generate(self, user_prompt, system_prompt, prompt_template, model,
                 temperature, max_tokens, lm_studio_url):

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

                final = prompt_template.replace("{prompt}", text)
                print(f"[NeuralBooru] {final[:160]}...")
                return (final,)

        except urllib.error.URLError as e:
            print(f"[NeuralBooru] Connection failed - is LM Studio running? ({e})")
            return (prompt_template.replace("{prompt}", user_prompt),)
        except Exception as e:
            print(f"[NeuralBooru] Error: {e}")
            return (prompt_template.replace("{prompt}", user_prompt),)


NODE_CLASS_MAPPINGS = {
    "NeuralBooru": NeuralBooru,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralBooru": "Neural Booru",
}
