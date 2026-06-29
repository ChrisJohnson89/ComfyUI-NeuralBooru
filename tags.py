"""
NeuralBooru tag validation.

Validates LLM-proposed tags against the real Danbooru tag vocabulary so the
node emits tags the model was actually trained on, not plausible-looking
natural language. The LLM proposes, this whitelist disposes.

Tag data: data/danbooru.csv, sourced from
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete
Format per row: tag, category, post_count, "alias1,alias2,..."
"""

import csv
import os
import re
import difflib

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "danbooru.csv")

# Danbooru category codes
CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

_WS = re.compile(r"\s+")
_HAS_LETTER = re.compile(r"[a-z]")


def normalize(tag):
    """Canonical form for matching: lowercase, unescaped, underscored."""
    t = tag.strip().lower()
    t = t.replace("\\(", "(").replace("\\)", ")")
    t = _WS.sub(" ", t).strip()
    t = t.replace(" ", "_")
    return t


def _morph_variants(norm):
    """Candidate stems for a word that did not resolve directly.

    Handles common verb/plural forms: smirking->smirk, posing->pose,
    crossed->cross, cats->cat. Conservative: only the last token is stemmed
    and very short stems are skipped to avoid false hits.
    """
    out = []
    if "_" in norm:  # only stem the final word of a phrase
        head, _, tail = norm.rpartition("_")
        prefix = head + "_"
    else:
        prefix, tail = "", norm
    for suf, repl in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
                      ("es", ""), ("s", "")):
        if tail.endswith(suf) and len(tail) - len(suf) >= 3:
            out.append(prefix + tail[:-len(suf)] + repl)
    return out


def to_prompt(tag):
    """Render a canonical tag for the prompt: spaces, escaped parens.

    Emoticon-style tags (no letters, e.g. >_<) keep their underscores.
    """
    if _HAS_LETTER.search(tag):
        t = tag.replace("_", " ")
    else:
        t = tag
    return t.replace("(", "\\(").replace(")", "\\)")


class TagDB:
    """Loads the Danbooru tag vocabulary and resolves candidate tags."""

    def __init__(self, path=_DATA_PATH):
        # normalized canonical tag -> (original_tag, category, post_count)
        self.canonical = {}
        # normalized alias -> normalized canonical tag
        self.alias = {}
        self._keys = None  # cached list for fuzzy matching
        self._load(path)

    def _load(self, path):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                tag = row[0].strip()
                category = int(row[1]) if len(row) > 1 and row[1].strip().isdigit() else 0
                count = int(row[2]) if len(row) > 2 and row[2].strip().isdigit() else 0
                norm = normalize(tag)
                if not norm:
                    continue
                self.canonical[norm] = (tag, category, count)
                if len(row) > 3 and row[3]:
                    for a in row[3].split(","):
                        a = normalize(a)
                        # real tags win over aliases; first alias wins ties
                        if a and a not in self.canonical and a not in self.alias:
                            self.alias[a] = norm

    def __len__(self):
        return len(self.canonical)

    def resolve(self, candidate, fuzzy_cutoff=0.0):
        """Return the normalized canonical tag for a candidate, or None.

        Tries exact match, then alias remap, then optional fuzzy match.
        """
        norm = normalize(candidate)
        if not norm:
            return None
        direct = self._lookup(norm)
        if direct:
            return direct
        # morphological variants: smirking -> smirk, posing -> pose, cats -> cat
        for v in _morph_variants(norm):
            hit = self._lookup(v)
            if hit:
                return hit
        if fuzzy_cutoff and fuzzy_cutoff > 0:
            if self._keys is None:
                self._keys = list(self.canonical.keys())
            hit = difflib.get_close_matches(norm, self._keys, n=1, cutoff=fuzzy_cutoff)
            if hit:
                return hit[0]
        return None

    def _lookup(self, norm):
        """Exact canonical or alias hit for an already-normalized string."""
        if norm in self.canonical:
            return norm
        if norm in self.alias:
            return self.alias[norm]
        return None

    def extract(self, candidate, fuzzy_cutoff=0.0):
        """Pull real tags out of a multi-word candidate that did not resolve.

        Greedy longest-match left to right, so "black crop top" yields
        ["black", "crop top"] and "dark classroom" yields ["classroom"].
        Single-word candidates return [] (nothing to salvage).
        """
        toks = [w for w in re.split(r"\s+", candidate.strip().lower()) if w]
        if len(toks) < 2:
            return []
        found, i = [], 0
        while i < len(toks):
            matched = False
            for j in range(len(toks), i, -1):
                sub = "_".join(toks[i:j])
                hit = self.resolve(sub, fuzzy_cutoff) if j - i > 1 else self._lookup(sub)
                if hit:
                    found.append(hit)
                    i = j
                    matched = True
                    break
            if not matched:
                i += 1
        return found

    def validate(self, text, strict=True, fuzzy_cutoff=0.0,
                 min_post_count=0, max_tags=0, exclude_categories=None):
        """Validate a comma-separated tag string.

        Returns (prompt_string, kept_tags, dropped_tags).
        - strict: drop candidates that resolve to no real tag (else keep raw)
        - fuzzy_cutoff: 0 disables fuzzy remapping
        - min_post_count: drop resolved tags rarer than this
        - max_tags: 0 means no limit
        - exclude_categories: iterable of category ints to drop (e.g. {1, 5})
        """
        exclude = set(exclude_categories or ())
        kept, dropped, seen = [], [], set()

        def accept(norm):
            """Apply category/count/dedupe filters; append if it passes."""
            _orig, category, count = self.canonical[norm]
            if category in exclude:
                return False
            if min_post_count and count < min_post_count:
                return False
            if norm in seen:
                return True  # already have it; treat as accepted (not dropped)
            seen.add(norm)
            kept.append(to_prompt(norm))
            return True

        for cand in (c.strip() for c in text.split(",")):
            if not cand:
                continue
            if max_tags and len(kept) >= max_tags:
                break

            norm = self.resolve(cand, fuzzy_cutoff)
            if norm is not None:
                if not accept(norm):
                    dropped.append(cand)
                continue

            # salvage real tags embedded in a multi-word phrase before dropping
            subs = self.extract(cand, fuzzy_cutoff)
            if subs:
                # evaluate every sub-tag (no short-circuit) so we keep all of them
                results = [accept(s) for s in subs]
                if any(results):
                    continue

            if strict:
                dropped.append(cand)
            else:
                key = normalize(cand)
                if key not in seen:
                    seen.add(key)
                    kept.append(to_prompt(key))

        return ", ".join(kept), kept, dropped


_DB = None


def get_db():
    """Lazy singleton. Returns None if the tag database is unavailable."""
    global _DB
    if _DB is None:
        try:
            _DB = TagDB()
        except Exception as e:
            print(f"[NeuralBooru] Tag database unavailable, validation disabled: {e}")
            _DB = False
    return _DB or None
