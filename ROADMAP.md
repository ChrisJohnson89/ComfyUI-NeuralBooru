# Roadmap

Planned work, roughly in order. PRs and discussion welcome.

## Node improvements

- **Better fuzzy matching.** Replace `difflib` with an optional
  [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) import (falling back to
  stdlib when absent, keeping the node dependency-free). Goal: make
  `fuzzy_cutoff` good enough to enable by default.
- **Tag DB refresh script.** `data/danbooru.csv` is a snapshot from
  [tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete).
  Add a script that re-downloads it and reports the delta (new tags, changed
  post counts) so the vocabulary can be kept current.
- **On-node result display.** Small JS web extension to show the final prompt
  and dropped tags directly on the node after execution, instead of only in
  the console.

## Finetuned drop-in tagger (see [finetune/README.md](finetune/README.md))

- **Phase 2: eval scorecard.** In-vocab rate, validator-change rate, and
  P/R/F1 against held-out pairs, so a finetuned model has to beat the
  prompted baseline before it ships.
- **Phase 3: scale with back-translation.** Generate English descriptions
  from real danbooru2023 tag sets (~100k pairs) so ground truth is real tags,
  not model output. Move base to Qwen2.5-1.5B.
- **Phase 4: release.** GGUF export and a HuggingFace model card with full
  data provenance.

## Considered and parked

- Tag implication pruning (needs implication data the CSV does not carry)
- Negative prompt generation (better handled by static templates)
- Multi-candidate generation (seed re-roll covers it)
