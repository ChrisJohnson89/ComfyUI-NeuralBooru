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
        if norm in self.canonical:
            return norm
        if norm in self.alias:
            return self.alias[norm]
        if fuzzy_cutoff and fuzzy_cutoff > 0:
            if self._keys is None:
                self._keys = list(self.canonical.keys())
            hit = difflib.get_close_matches(norm, self._keys, n=1, cutoff=fuzzy_cutoff)
            if hit:
                return hit[0]
        return None

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

        for cand in (c.strip() for c in text.split(",")):
            if not cand:
                continue
            norm = self.resolve(cand, fuzzy_cutoff)
            if norm is None:
                if strict:
                    dropped.append(cand)
                    continue
                # lenient: keep the raw candidate as-is
                key = normalize(cand)
                if key in seen:
                    continue
                seen.add(key)
                kept.append(to_prompt(key))
                continue

            original, category, count = self.canonical[norm]
            if category in exclude:
                dropped.append(cand)
                continue
            if min_post_count and count < min_post_count:
                dropped.append(cand)
                continue
            if norm in seen:
                continue
            seen.add(norm)
            kept.append(to_prompt(norm))
            if max_tags and len(kept) >= max_tags:
                break

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
