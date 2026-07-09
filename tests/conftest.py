"""Shared fixtures: load the node module (repo root is not an importable
package name, so it is loaded by path) and the tag DB once per session."""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def node_module():
    spec = importlib.util.spec_from_file_location(
        "neuralbooru", os.path.join(ROOT, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neuralbooru"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def db():
    import tags
    d = tags.get_db()
    assert d is not None, "danbooru.csv failed to load"
    return d


@pytest.fixture(scope="session")
def llm_module(node_module):
    spec = importlib.util.spec_from_file_location(
        "llm_node", os.path.join(ROOT, "llm_node.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def validator_module(node_module):
    spec = importlib.util.spec_from_file_location(
        "validator_node", os.path.join(ROOT, "validator_node.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
