"""
Generate diverse English scene descriptions to feed the data factory.

Phase 1 uses a combinatorial generator (no LLM, fully offline and seedable)
so we get reliable variety and reproducibility while proving the loop. The
goal is realistic *user-style* prompts: the kind of thing someone types into
NeuralBooru, ranging from terse to verbose.

Usage:
    python finetune/gen_prompts.py --count 2000 --seed 7 --out finetune/data/prompts.txt
"""

import argparse
import os
import random
import re

SUBJECTS = [
    "a girl", "a young woman", "a teenage girl", "a woman", "a boy",
    "a young man", "a man", "two girls", "two women", "a couple",
    "a group of friends", "an elf girl", "a catgirl", "a witch",
    "a knight", "a samurai", "a magical girl", "a swordswoman",
    "a schoolgirl", "a princess", "a robot girl", "an angel", "a demon girl",
]

HAIR = [
    "long black hair", "short blonde hair", "silver twin tails",
    "messy brown hair", "pink bob cut", "blue ponytail", "red curly hair",
    "white long hair", "green braids", "purple short hair", "",
]

EYES = [
    "blue eyes", "green eyes", "red eyes", "golden eyes", "heterochromia",
    "purple eyes", "brown eyes", "",
]

EXPRESSION = [
    "smiling", "looking serious", "blushing", "crying", "laughing",
    "with a calm expression", "smirking", "looking surprised", "",
]

CLOTHING = [
    "in a red dress", "wearing a school uniform", "in armor",
    "in a kimono", "wearing a hoodie", "in a maid outfit",
    "in casual clothes", "wearing a long coat", "in a swimsuit",
    "in a gothic lolita dress", "in a business suit", "in a witch hat and robe",
]

SETTING = [
    "on a beach", "in a forest", "in a futuristic city", "in a classroom",
    "in a cozy cafe", "on a rooftop", "in a snowy field", "in a flower garden",
    "in an old library", "on a busy street", "in a bedroom", "in a shrine",
    "on a mountain trail", "in a spaceship", "in a rainy alley",
]

TIME = [
    "at sunset", "at night", "in the morning", "at dawn", "under a full moon",
    "on a cloudy day", "during a storm", "at golden hour", "",
]

ACTION = [
    "sitting", "standing", "walking", "running", "holding a sword",
    "reading a book", "drinking coffee", "casting a spell", "dancing",
    "looking at the viewer", "gazing into the distance", "",
]

MOOD = [
    "peaceful and warm", "dramatic and moody", "cheerful", "melancholic",
    "mysterious", "energetic", "dreamy and soft", "tense", "",
]

STYLE = [
    "anime style", "detailed illustration", "cinematic lighting",
    "soft watercolor look", "vibrant colors", "",
]


def _pick(rng, options):
    return rng.choice(options)


def _join(parts):
    s = " ".join(p for p in parts if p).strip()
    # tidy comma spacing produced by inline "," tokens: "hoodie , sitting" -> "hoodie, sitting"
    return re.sub(r"\s+,", ",", s)


# Phrasing templates of varying length/structure, from terse to verbose.
def _templates(rng):
    subj = _pick(rng, SUBJECTS)
    hair = _pick(rng, HAIR)
    eyes = _pick(rng, EYES)
    expr = _pick(rng, EXPRESSION)
    cloth = _pick(rng, CLOTHING)
    setting = _pick(rng, SETTING)
    time = _pick(rng, TIME)
    action = _pick(rng, ACTION)
    mood = _pick(rng, MOOD)
    style = _pick(rng, STYLE)

    desc = _join([hair, "and" if hair and eyes else "", eyes])
    looks = _join([d for d in [desc, expr] if d])

    candidates = [
        # terse
        _join([subj, cloth, setting]),
        _join([subj, "with", looks]) if looks else _join([subj, setting]),
        # medium
        _join([subj, "with", looks, cloth, setting, time]),
        _join([subj, cloth, action, setting, time]),
        # verbose / descriptive
        _join([subj, "with", looks, cloth, ",", action, setting, time, ",", mood]),
        _join([f"{style},", subj, "with", looks, cloth, setting, time]),
        # mood-led (forces the model to infer, not copy)
        _join([f"a {mood} scene of", subj, cloth, setting, time]) if mood else
        _join([subj, cloth, setting, time]),
    ]
    return [c for c in candidates if c and len(c.split()) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="finetune/data/prompts.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen, out = set(), []
    # cap attempts so we never spin forever once the combinatorial space thins
    attempts, max_attempts = 0, args.count * 50
    while len(out) < args.count and attempts < max_attempts:
        attempts += 1
        line = rng.choice(_templates(rng))
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {len(out)} unique prompts to {args.out} "
          f"(requested {args.count}, {attempts} attempts)")


if __name__ == "__main__":
    main()
