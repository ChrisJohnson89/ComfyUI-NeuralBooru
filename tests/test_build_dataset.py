"""Tests for the finetune dataset builder's split and cache integrity."""

import json
import os
import sys

import pytest

FINETUNE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "finetune")


@pytest.fixture(scope="session")
def bd():
    if FINETUNE not in sys.path:
        sys.path.insert(0, FINETUNE)
    import build_dataset
    return build_dataset


def test_split_is_stable_as_cache_grows(bd):
    prompts = [f"prompt {i} about a scene" for i in range(1000)]
    first = {p: bd.split_of(p, 7, 0.05, 0.05) for p in prompts[:200]}
    after = {p: bd.split_of(p, 7, 0.05, 0.05) for p in prompts}
    assert all(after[p] == first[p] for p in first)


def test_split_fractions_roughly_honored(bd):
    prompts = [f"prompt {i} about a scene" for i in range(2000)]
    splits = [bd.split_of(p, 7, 0.05, 0.05) for p in prompts]
    assert 0.03 < splits.count("test") / len(splits) < 0.07
    assert 0.03 < splits.count("valid") / len(splits) < 0.07


def test_split_seed_redeals(bd):
    prompts = [f"prompt {i}" for i in range(500)]
    a = [bd.split_of(p, 7, 0.05, 0.05) for p in prompts]
    b = [bd.split_of(p, 8, 0.05, 0.05) for p in prompts]
    assert a != b


def test_load_cache_filters_stale_system_prompt(bd, tmp_path):
    cur = bd.sys_hash("current system prompt")
    path = tmp_path / "cache.jsonl"
    rows = [
        {"prompt": "a", "clean": "1girl", "n_kept": 5, "sys": cur},
        {"prompt": "b", "clean": "1boy", "n_kept": 5, "sys": "deadbeef0000"},
        {"prompt": "c", "clean": "cat", "n_kept": 5},  # unstamped -> current
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\nnot json\n",
        encoding="utf-8")
    done, stale = bd.load_cache(str(path), cur)
    assert set(done) == {"a", "c"}
    assert stale == 1


def test_write_split_chat_format(bd, tmp_path):
    records = [{"prompt": f"prompt {i}", "clean": "1girl, beach"}
               for i in range(50)]
    n_train, n_val, n_test = bd.write_split(
        records, str(tmp_path), 0.1, 0.1, 7, "SYSTEM")
    assert n_train + n_val + n_test == 50
    row = json.loads(
        (tmp_path / "train.jsonl").read_text().splitlines()[0])
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][0]["content"] == "SYSTEM"
