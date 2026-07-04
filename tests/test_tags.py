"""Tests for the Danbooru tag validator (tags.py)."""

import tags


# --- normalize / to_prompt ------------------------------------------------

def test_normalize_basic():
    assert tags.normalize("  Blue   Eyes ") == "blue_eyes"
    assert tags.normalize("hatsune miku") == "hatsune_miku"


def test_normalize_unescapes_parens():
    assert tags.normalize("hatsune miku \\(vocaloid\\)") == "hatsune_miku_(vocaloid)"


def test_to_prompt_spaces_and_escaping():
    assert tags.to_prompt("blue_eyes") == "blue eyes"
    assert tags.to_prompt("hatsune_miku_(vocaloid)") == "hatsune miku \\(vocaloid\\)"


def test_to_prompt_emoticons_keep_underscores():
    assert tags.to_prompt(">_<") == ">_<"


# --- resolve ----------------------------------------------------------------

def test_resolve_exact(db):
    assert db.resolve("1girl") == "1girl"
    assert db.resolve("blue eyes") == "blue_eyes"


def test_resolve_alias_remaps_to_canonical(db):
    # pick a real alias from the loaded DB so the test tracks the data file
    alias, canonical = next(iter(db.alias.items()))
    assert db.resolve(alias.replace("_", " ")) == canonical


def test_resolve_morphological_variants(db):
    assert db.resolve("smirking") == "smirk"
    assert db.resolve("cats") == "cat"


def test_resolve_singular_to_plural(db):
    assert db.resolve("blue eye") == "blue_eyes"
    assert db.resolve("golden eye") == "yellow_eyes"  # via golden_eyes alias


def test_resolve_merged_tag_aliases(db):
    # danbooru merged silver_hair into grey_hair; CSV lost the alias
    assert db.resolve("silver hair") == "grey_hair"
    assert db.resolve("golden hair") == "blonde_hair"


def test_resolve_trailing_s_typo_via_morph(db):
    # "eyess" is caught by the -s/-es stemmer, no fuzzy needed
    assert db.resolve("blue eyess") == "blue_eyes"


def test_resolve_fuzzy_off_by_default(db):
    assert db.resolve("blue eyez") is None


def test_resolve_fuzzy_when_enabled(db):
    assert db.resolve("blue eyez", fuzzy_cutoff=0.85) == "blue_eyes"


def test_resolve_garbage(db):
    assert db.resolve("qzxjwv nonsense phrase") is None
    assert db.resolve("") is None


# --- extract (multi-word salvage) ------------------------------------------

def test_extract_greedy_longest_match(db):
    # "crop top" must win over "crop" + "top"; "black" alone is not a tag
    assert db.extract("black crop top") == ["crop_top"]


def test_extract_all_words_salvageable(db):
    assert db.extract("dark classroom") == ["dark", "classroom"]


def test_extract_single_word_returns_nothing(db):
    assert db.extract("classroom") == []


# --- validate ----------------------------------------------------------------

def test_validate_strict_drops_junk(db):
    prompt, kept, dropped = db.validate("1girl, utter nonsense qzx, beach")
    assert kept == ["1girl", "beach"]
    assert dropped == ["utter nonsense qzx"]
    assert prompt == "1girl, beach"


def test_validate_non_strict_keeps_raw(db):
    prompt, kept, dropped = db.validate("1girl, qzx wibble", strict=False)
    assert "qzx wibble" in prompt
    assert dropped == []


def test_validate_dedupes(db):
    _, kept, dropped = db.validate("1girl, 1girl, blue eyes")
    assert kept == ["1girl", "blue eyes"]
    assert dropped == []


def test_validate_max_tags_simple(db):
    _, kept, _ = db.validate("1girl, beach, smile, ocean", max_tags=2)
    assert kept == ["1girl", "beach"]


def test_validate_max_tags_not_exceeded_by_salvage(db):
    # regression: the multi-word salvage path used to blow past max_tags
    _, kept, _ = db.validate(
        "1girl, black crop top wet hair smile, beach", max_tags=2)
    assert len(kept) == 2


def test_validate_min_post_count(db):
    # 1girl has millions of posts; pick something guaranteed above any bar
    _, kept, dropped = db.validate("1girl", min_post_count=1)
    assert kept == ["1girl"]
    count = db.canonical["1girl"][2]
    _, kept, dropped = db.validate("1girl", min_post_count=count + 1)
    assert kept == [] and dropped == ["1girl"]


def test_validate_exclude_categories(db):
    # highres is a meta tag (category 5)
    assert db.canonical["highres"][1] == 5
    _, kept, dropped = db.validate("1girl, highres", exclude_categories={5})
    assert kept == ["1girl"]
    assert dropped == ["highres"]


def test_validate_sort_tags_danbooru_order(db):
    # count > character > copyright > general > meta
    assert db.canonical["hatsune_miku"][1] == 4
    assert db.canonical["vocaloid"][1] == 3
    assert db.canonical["absurdres"][1] == 5
    _, kept, _ = db.validate(
        "night, absurdres, hatsune miku, vocaloid, 1girl", sort_tags=True)
    assert kept == ["1girl", "hatsune miku", "vocaloid", "night", "absurdres"]


def test_validate_unsorted_by_default(db):
    _, kept, _ = db.validate("night, 1girl")
    assert kept == ["night", "1girl"]


def test_validate_sort_stable_within_group(db):
    _, kept, _ = db.validate("beach, night, smile, 1girl", sort_tags=True)
    assert kept == ["1girl", "beach", "night", "smile"]


def test_validate_salvage_inside_full_line(db):
    prompt, kept, dropped = db.validate("1girl, black crop top, beach")
    assert kept == ["1girl", "crop top", "beach"]
    assert dropped == []
