# NeuralBooru finetuning

Tooling to distill the NeuralBooru pipeline (LLM proposes + validator cleans)
into a small standalone model that does English -> Danbooru tags in one step,
deployable through the existing node via LM Studio (MLX).

Scripts here are offline tooling and are **not** imported by the ComfyUI node.
Generated data under `data/` is gitignored.

## Phase 1 - prove the loop (pipeline as data factory)

```bash
# 1. Generate diverse English scene descriptions
python finetune/gen_prompts.py --count 8000 --seed 7 --out finetune/data/prompts.txt

# 2. With LM Studio serving the tagger (qwen/qwen3-1.7b), run them through
#    the node's exact pipeline (system prompt + validation). Resumable.
python finetune/build_dataset.py --prompts finetune/data/prompts.txt \
    --model qwen/qwen3-1.7b --out-dir finetune/data

# -> finetune/data/{train,valid,test}.jsonl in mlx-lm chat format
#    finetune/data/cache.jsonl keeps raw/kept/dropped for inspection
```

The dataset uses the node's **exact** `DEFAULT_SYSTEM_PROMPT` (loaded from
`../__init__.py`) so training matches inference.

### Train (mlx-lm LoRA, Apple Silicon, 16 GB)

mlx-lm lives in an isolated venv (Homebrew Python blocks system installs):

```bash
python3.11 -m venv finetune/.venv
finetune/.venv/bin/pip install -U mlx-lm
```

Then one command does convert (4-bit) -> LoRA train -> test -> fuse:

```bash
bash finetune/train.sh
# override any knob: ITERS=2000 NUM_LAYERS=-1 BATCH_SIZE=2 bash finetune/train.sh
```

`--mask-prompt` is on, so loss is computed only on the assistant tags, not the
fixed system prompt + input. The fused model lands in
`finetune/models/neuralbooru-0.5b-mlx`.

Then point LM Studio at that folder and run it through the NeuralBooru node
unchanged. Loop proven when tags come out and validate.

## Roadmap
- Phase 2: eval scorecard (P/R/F1 vs gold, in-vocab rate, validator-change rate)
- Phase 3: scale with back-translation from real danbooru2023 tag sets (~100k pairs), base -> Qwen2.5-1.5B
- Phase 4: GGUF export + HuggingFace model card (NSFW-capable, provenance)
