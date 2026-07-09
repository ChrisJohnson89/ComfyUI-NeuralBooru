"""
NeuralBooruValidator - Danbooru tag validation node.

Validates comma-separated tags against the real Danbooru vocabulary,
applies aliasing/sorting/fuzzy matching, and wraps the result in a
prompt template. Has no LLM awareness; works standalone with hand-typed
tags or wired after NeuralBooruLLM for the full original pipeline.

When wired in sequence:  NeuralBooruLLM.raw_tags -> NeuralBooruValidator.tags
the two nodes reproduce the original NeuralBooru node exactly.
"""

try:
    from . import tags as tagdb
    from . import (TEMPLATE_PRESETS, NOVA_ANIME_XL_TEMPLATE,
                   _apply_template, _parse_categories)
except ImportError:
    import tags as tagdb
    from neuralbooru import (TEMPLATE_PRESETS, NOVA_ANIME_XL_TEMPLATE,
                              _apply_template, _parse_categories)


class NeuralBooruValidator:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Comma-separated tags to validate. From NeuralBooruLLM "
                               "or typed manually."
                }),
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": NOVA_ANIME_XL_TEMPLATE,
                    "tooltip": "Final prompt wrapper; {prompt} is replaced with the "
                               "validated tags. If {prompt} is missing, tags are appended."
                }),
            },
            "optional": {
                "is_fallback": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True (set by NeuralBooruLLM on error), skip "
                               "validation and pass tags through into the template. "
                               "Reproduces the original node's use_input_text fallback."
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
                "exclude_categories": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Tag categories to drop, comma-separated: "
                               "artist, character, copyright, meta, general."
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

    def generate(self, tags, prompt_template,
                 is_fallback=False, validate_tags=True, strict_tags=True,
                 fuzzy_cutoff=0.0, min_post_count=0, max_tags=0,
                 exclude_categories="", template_preset="custom",
                 sort_tags=True):

        template = TEMPLATE_PRESETS.get(template_preset, prompt_template)

        # Error fallback or validation disabled: pass through into template
        if is_fallback or not validate_tags:
            final = _apply_template(template, tags)
            print(f"[NeuralBooruValidator] {final[:160]}...")
            return (final, "", tags)

        db = tagdb.get_db()
        if db is None:
            # Tag database unavailable; pass through
            final = _apply_template(template, tags)
            print(f"[NeuralBooruValidator] {final[:160]}...")
            return (final, "", tags)

        validated, _kept, dropped = db.validate(
            tags,
            strict=strict_tags,
            fuzzy_cutoff=fuzzy_cutoff,
            min_post_count=min_post_count,
            max_tags=max_tags,
            exclude_categories=_parse_categories(exclude_categories),
            sort_tags=sort_tags,
        )

        final = _apply_template(template, validated)
        print(f"[NeuralBooruValidator] {final[:160]}...")
        if dropped:
            print(f"[NeuralBooruValidator] dropped {len(dropped)} non-tags: {dropped}")
        return (final, ", ".join(dropped), validated)
