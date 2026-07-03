#!/usr/bin/env bash
#
# NeuralBooru Phase 1 training: LoRA-finetune a small Qwen on the
# English -> Danbooru-tags dataset, then fuse to a standalone MLX model
# that LM Studio loads and the node uses unchanged.
#
# Prereqs: finetune/.venv with mlx-lm, and finetune/data/{train,valid,test}.jsonl
# (produced by build_dataset.py).
#
# Usage:
#   bash finetune/train.sh
# Override any knob via env, e.g.:
#   ITERS=2000 NUM_LAYERS=-1 BATCH_SIZE=2 bash finetune/train.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=finetune/.venv/bin/python
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MLX_BASE="${MLX_BASE:-finetune/models/qwen2.5-0.5b-4bit}"
FUSED="${FUSED:-finetune/models/neuralbooru-0.5b-mlx}"
DATA="${DATA:-finetune/data}"
ADAPTERS="${ADAPTERS:-finetune/adapters}"

# 16 GB-safe training defaults (0.5B @ 4-bit). Tune via env.
NUM_LAYERS="${NUM_LAYERS:-16}"       # -1 = all layers (fine for 0.5B)
BATCH_SIZE="${BATCH_SIZE:-1}"
ITERS="${ITERS:-1000}"               # minibatches; scale with dataset size
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-200}"

echo "==> 1/4 Convert base to 4-bit MLX ($BASE_MODEL)"
if [ -d "$MLX_BASE" ]; then
  echo "    $MLX_BASE exists, skipping convert"
else
  $PY -m mlx_lm convert --hf-path "$BASE_MODEL" -q --q-bits 4 --mlx-path "$MLX_BASE"
fi

echo "==> 2/4 LoRA train (num-layers=$NUM_LAYERS batch=$BATCH_SIZE iters=$ITERS)"
# --mask-prompt: only the assistant tags contribute to the loss, not the
# fixed system prompt + user input. That is what we want the model to learn.
$PY -m mlx_lm lora \
  --model "$MLX_BASE" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers "$NUM_LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --iters "$ITERS" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --learning-rate "$LEARNING_RATE" \
  --steps-per-eval "$STEPS_PER_EVAL" \
  --val-batches -1 \
  --adapter-path "$ADAPTERS"

echo "==> 3/4 Evaluate on held-out test set"
$PY -m mlx_lm lora \
  --model "$MLX_BASE" \
  --data "$DATA" \
  --adapter-path "$ADAPTERS" \
  --test || echo "    (test split may be empty at small scale; skipping)"

echo "==> 4/4 Fuse adapters into standalone MLX model -> $FUSED"
$PY -m mlx_lm fuse \
  --model "$MLX_BASE" \
  --adapter-path "$ADAPTERS" \
  --save-path "$FUSED"

echo
echo "Done. Point LM Studio at: $FUSED"
echo "Then in the NeuralBooru node set model to the loaded id and run unchanged."
echo "Quick CLI sanity check:"
echo "  $PY -m mlx_lm generate --model $FUSED \\"
echo "     --system-prompt \"\$(python3 -c 'import finetune.pipeline as p; print(p.DEFAULT_SYSTEM_PROMPT)')\" \\"
echo "     --prompt 'a lone samurai in the rain at night' --max-tokens 120"
