"""Tests for utils.helpers.clean_text."""
import importlib.util
from pathlib import Path

HELPERS = Path(__file__).resolve().parents[1] / "src" / "utils" / "helpers.py"
spec = importlib.util.spec_from_file_location("helpers", HELPERS)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_clean_text_returns_empty_string_for_none():
    assert module.clean_text(None) == ""


def test_clean_text_still_strips_urls_and_tags():
    assert "example.com" not in module.clean_text("<p>see http://example.com now</p>").lower()
