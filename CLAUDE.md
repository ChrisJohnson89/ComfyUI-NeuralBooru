# CLAUDE.md

ComfyUI custom node: English scene description -> LLM -> validated Danbooru
tags -> templated prompt. Works with any OpenAI-compatible server; Ollama is
auto-detected and driven through its native API instead.

## Layout

- `__init__.py` - the node (single class `NeuralBooru`), HTTP client, template
  presets. Loaded by ComfyUI as a package; also importable standalone.
- `tags.py` - tag validator (exact -> alias -> stemming -> sub-phrase salvage
  -> optional fuzzy -> drop, then optional Danbooru-order sort). Pure,
  dependency-free, and the most test-covered code here.
- `data/danbooru.csv` - ~140k tag vocabulary (tag, category, post_count,
  aliases). Required at runtime.
- `tests/` - pytest suite. The node is tested against a mocked
  `urllib.request.urlopen`; no server needed.
- `finetune/` - offline dataset/training tooling. NOT imported by the node.
  It loads `DEFAULT_SYSTEM_PROMPT` from `__init__.py` by path so training
  always matches inference. Generated data under `finetune/data/` is
  gitignored.
- `workflows/AI_Anime.json` - example workflow shipped to users.

## Hard constraints

- **The node stays pure stdlib.** Zero dependencies is a shipped feature.
  Never add imports beyond the standard library to `__init__.py` or `tags.py`.
- **ComfyUI restores `widgets_values` by position.** Never insert or reorder
  entries in `INPUT_TYPES`; add new inputs at the END of the `optional` group,
  and update `workflows/AI_Anime.json`'s widget list in the same commit.
- **`DEFAULT_SYSTEM_PROMPT` is a single source of truth.** The finetune cache
  stamps records with its hash; editing it invalidates cached training pairs
  (by design). Do not duplicate the prompt anywhere.

## Testing

```bash
python3 -m pytest tests/ -q   # needs pytest; code itself is stdlib-only
```

All tests must pass before pushing; CI runs them on every push and PR.
When touching validator behavior, encode the true behavior in a test; the
Danbooru vocabulary has surprises (e.g. "dark" is a tag, "black" is not).

## Releases (ComfyUI Registry + GitHub)

- Bump `version` in `pyproject.toml` on main -> triggers the registry publish
  workflow. Push a `v*` tag -> triggers the GitHub release workflow.
- Registry versions are IMMUTABLE and version **1.2.0 is permanently burned**
  (an un-shipped build); never reuse a published number, even a deleted one.
- After publishing, verify the scan verdict; a version can sit in
  `NodeVersionStatusFlagged` and the registry silently serves the previous
  Active version:
  `curl -s https://api.comfy.org/nodes/comfyui-neuralbooru/versions`
- `.comfyignore` controls the published archive. `finetune/`, `tests/`, and
  `.github/` are excluded; keep the shipped package to node runtime files
  (a slim package also avoids the registry's opaque security scanner).

## Server quirks worth remembering

- Ollama's OpenAI-compatible endpoint ignores thinking control; that is why
  `generate()` probes `/api/version` and uses native `/api/chat` with
  `think` when it finds Ollama. Models without a thinking mode 400 on the
  flag; the node retries once without it.
- Reasoning models may return output in `reasoning`/`reasoning_content`
  (OpenAI-style) or `thinking` (Ollama native) with empty `content`; the
  node falls back through those fields, then think-strips and validates.
