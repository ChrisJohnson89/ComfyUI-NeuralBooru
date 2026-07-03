"""
Build the English -> Danbooru-tags training set (Phase 1, pipeline-as-factory).

For each English prompt: call the tagger model through LM Studio with the
node's exact DEFAULT_SYSTEM_PROMPT, strip <think>, validate against the
Danbooru vocab, and keep the cleaned (prompt -> tags) pair. Output is
mlx-lm chat format so training matches how the node calls the model.

Resumable: every processed prompt is appended to a raw cache (cache.jsonl)
with its raw output / kept / dropped tags. Re-running skips prompts already
in the cache, so an interrupted overnight run picks up where it left off.

The train/valid/test split is rebuilt from the cache each run, but assignment
is a stable hash of (seed, prompt): a prompt keeps its split forever, so
held-out samples never migrate into training as the cache grows.

Each cache record is stamped with a hash of the system prompt it was
generated under. Records from an older system prompt are treated as stale:
excluded from the dataset and reprocessed. Unstamped records (pre-dating
stamping) are assumed current.

Usage:
    # 1) make prompts
    python finetune/gen_prompts.py --count 2000 --out finetune/data/prompts.txt
    # 2) make sure LM Studio is serving the tagger model, then:
    python finetune/build_dataset.py --prompts finetune/data/prompts.txt \
        --model qwen/qwen3-1.7b --out-dir finetune/data
"""

import argparse
import hashlib
import json
import os

import pipeline


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def sys_hash(system):
    """Short fingerprint of the system prompt, stamped on cache records."""
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]


def load_cache(path, current_sys):
    """Return ({prompt: record}, stale_count) for already-processed prompts.

    Records stamped with a different system-prompt hash are stale: skipped
    here so they get reprocessed and never mix two prompt regimes in one
    dataset. Unstamped records (pre-dating stamping) are assumed current.
    """
    done, stale = {}, 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    if rec.get("sys", current_sys) != current_sys:
                        stale += 1
                        continue
                    done[rec["prompt"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return done, stale


def split_of(prompt, seed, val_frac, test_frac):
    """Stable split assignment by hash of (seed, prompt).

    A prompt keeps its split for the life of the project (per seed), so
    held-out samples never leak into training as the cache grows between
    runs. Changing --seed re-deals the whole partition.
    """
    h = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / 2 ** 64
    if x < test_frac:
        return "test"
    if x < test_frac + val_frac:
        return "valid"
    return "train"


def write_split(records, out_dir, val_frac, test_frac, seed, system):
    """Write train/valid/test JSONL in chat format, split by stable hash."""
    buckets = {"train": [], "valid": [], "test": []}
    for r in records:
        buckets[split_of(r["prompt"], seed, val_frac, test_frac)].append(r)

    def dump(name, rows):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                obj = {"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["clean"]},
                ]}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return path

    dump("train.jsonl", buckets["train"])
    dump("valid.jsonl", buckets["valid"])
    dump("test.jsonl", buckets["test"])
    return len(buckets["train"]), len(buckets["valid"]), len(buckets["test"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="finetune/data/prompts.txt")
    ap.add_argument("--out-dir", default="finetune/data")
    ap.add_argument("--model", default="qwen/qwen3-1.7b")
    ap.add_argument("--url", default="http://localhost:1234")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--min-tags", type=int, default=4,
                    help="drop pairs whose validated output has fewer tags")
    ap.add_argument("--fuzzy-cutoff", type=float, default=0.0)
    ap.add_argument("--min-post-count", type=int, default=0)
    ap.add_argument("--max-tags", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="process at most N new prompts (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "cache.jsonl")

    current_sys = sys_hash(pipeline.DEFAULT_SYSTEM_PROMPT)
    prompts = read_prompts(args.prompts)
    done, stale = load_cache(cache_path, current_sys)
    todo = [p for p in prompts if p not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(prompts)} prompts | {len(done)} cached | {len(todo)} to process")
    if stale:
        print(f"NOTE: {stale} cache records were generated under a different "
              f"system prompt; excluded and queued for reprocessing.")

    lm = pipeline.LMStudio(args.url, timeout=args.max_tokens // 2 + 60)
    if todo:
        loaded = lm.models()
        if not loaded:
            raise SystemExit(
                f"Could not reach LM Studio at {args.url}. Start it and load a model."
            )
        if args.model not in loaded:
            print(f"WARNING: model '{args.model}' not in loaded models {loaded}. "
                  f"LM Studio may route to whatever is loaded.")

    processed = errors = 0
    with open(cache_path, "a", encoding="utf-8") as cache:
        for i, prompt in enumerate(todo, 1):
            try:
                raw = lm.chat(args.model, pipeline.DEFAULT_SYSTEM_PROMPT, prompt,
                              temperature=args.temperature, max_tokens=args.max_tokens)
                clean, kept, dropped = pipeline.validate(
                    raw, strict=True, fuzzy_cutoff=args.fuzzy_cutoff,
                    min_post_count=args.min_post_count, max_tags=args.max_tags)
                rec = {"prompt": prompt, "raw": raw, "clean": clean,
                       "kept": kept, "dropped": dropped, "n_kept": len(kept),
                       "sys": current_sys}
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
                done[prompt] = rec
                processed += 1
            except Exception as e:  # keep going; one bad call shouldn't kill an overnight run
                errors += 1
                print(f"  [{i}] error on {prompt[:50]!r}: {e}")
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} processed ({errors} errors)")

    # build split from all cached records that clear the quality bar
    good = [r for r in done.values() if r.get("n_kept", 0) >= args.min_tags]
    n_train, n_val, n_test = write_split(
        good, args.out_dir, args.val_frac, args.test_frac, args.seed,
        pipeline.DEFAULT_SYSTEM_PROMPT)

    dropped_low = len(done) - len(good)
    print(f"\nThis run: {processed} new, {errors} errors.")
    print(f"Cache total: {len(done)} | usable (>= {args.min_tags} tags): {len(good)} "
          f"| below bar: {dropped_low}")
    print(f"Split -> train {n_train} | valid {n_val} | test {n_test}  in {args.out_dir}/")


if __name__ == "__main__":
    main()
