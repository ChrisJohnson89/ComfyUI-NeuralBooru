"""Tests for the NeuralBooruValidator node (validator_node.py)."""

import pytest


def base_kwargs(validator_module):
    return dict(
        tags="1girl, beach, smile",
        prompt_template="{prompt}",
    )


@pytest.fixture
def call(validator_module):
    """Run NeuralBooruValidator.generate; returns (prompt, dropped, tags)."""
    def _call(**overrides):
        kwargs = base_kwargs(validator_module)
        kwargs.update(overrides)
        return validator_module.NeuralBooruValidator().generate(**kwargs)
    return _call


# --- basic validation --------------------------------------------------------

def test_validates_and_returns_prompt(call):
    prompt, dropped, tags = call(tags="1girl, complete gibberish qzx, beach")
    assert prompt == "1girl, beach"
    assert dropped == "complete gibberish qzx"
    assert tags == "1girl, beach"


def test_drops_junk_reports_dropped(call):
    _, dropped, _ = call(tags="1girl, qzx wibble, beach")
    assert "qzx wibble" in dropped


def test_passes_through_when_validate_tags_off(call):
    prompt, dropped, tags = call(
        tags="1girl, qzx wibble, beach", validate_tags=False)
    assert prompt == "1girl, qzx wibble, beach"
    assert dropped == ""
    assert tags == "1girl, qzx wibble, beach"


# --- is_fallback -------------------------------------------------------------

def test_is_fallback_skips_validation(call):
    prompt, dropped, tags = call(
        tags="a girl on a beach", is_fallback=True)
    assert prompt == "a girl on a beach"
    assert dropped == ""
    assert tags == "a girl on a beach"


def test_is_fallback_with_template(call):
    prompt, dropped, tags = call(
        tags="a girl on a beach",
        is_fallback=True,
        prompt_template="quality, {prompt}")
    assert prompt == "quality, a girl on a beach"
    assert dropped == ""
    assert tags == "a girl on a beach"


# --- template ----------------------------------------------------------------

def test_template_placeholder(call):
    prompt, *_ = call(prompt_template="quality, {prompt}, BREAK")
    assert prompt == "quality, 1girl, beach, smile, BREAK"


def test_template_without_placeholder_appends(call):
    prompt, *_ = call(prompt_template="masterpiece, best quality")
    assert prompt == "masterpiece, best quality, 1girl, beach, smile"


def test_template_preset_overrides_field(validator_module):
    prompt, *_ = validator_module.NeuralBooruValidator().generate(
        tags="1girl, beach",
        prompt_template="ignored {prompt}",
        template_preset="pony")
    assert prompt.startswith("score_9, score_8_up")
    assert "ignored" not in prompt


def test_template_preset_custom_uses_field(call):
    prompt, *_ = call(template_preset="custom", prompt_template="X, {prompt}")
    assert prompt == "X, 1girl, beach, smile"


def test_all_presets_have_placeholder(validator_module):
    for name, template in validator_module.TEMPLATE_PRESETS.items():
        assert "{prompt}" in template, name


# --- tag filtering options ---------------------------------------------------

def test_exclude_categories(call):
    prompt, dropped, _ = call(
        tags="1girl, highres", exclude_categories="meta")
    assert prompt == "1girl"
    assert dropped == "highres"


def test_sort_tags_on_by_default(call):
    prompt, *_ = call(tags="night, hatsune miku, 1girl")
    assert prompt == "1girl, hatsune miku, night"


def test_sort_tags_off_preserves_order(call):
    prompt, *_ = call(tags="night, 1girl", sort_tags=False)
    assert prompt == "night, 1girl"


def test_tags_output_is_untemplated(call):
    prompt, _, tags = call(prompt_template="quality, {prompt}, BREAK")
    assert prompt == "quality, 1girl, beach, smile, BREAK"
    assert tags == "1girl, beach, smile"


def test_max_tags(call):
    prompt, _, _ = call(tags="1girl, beach, smile, ocean", max_tags=2)
    assert prompt == "1girl, beach"


def test_min_post_count(call):
    prompt, dropped, _ = call(tags="1girl", min_post_count=9999999)
    assert prompt == ""
    assert dropped == "1girl"


# --- parse_categories --------------------------------------------------------

def test_parse_categories(validator_module):
    assert validator_module._parse_categories("artist, meta, 4") == {1, 5, 4}
    assert validator_module._parse_categories("") == set()
    assert validator_module._parse_categories("bogus") == set()
