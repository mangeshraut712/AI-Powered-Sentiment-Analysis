"""Tests for label consolidation and input guards in src/preprocessing.py."""

import pandas as pd

from src.preprocessing import clean_text, consolidate_sentiments


def make_frame(sentiments):
    return pd.DataFrame(
        {
            "content": [f"message {i}" for i in range(len(sentiments))],
            "sentiment": list(sentiments),
        }
    )


def test_consolidate_sentiments_applies_the_default_thirteen_to_seven_mapping():
    df = make_frame(["hate", "fun", "boredom", "worry", "happiness", "neutral"])

    result = consolidate_sentiments(df)

    # "hate" folds into anger, "fun" into happiness, "boredom" into neutral;
    # already-canonical labels must pass through unchanged.
    assert result["sentiment"].tolist() == [
        "anger",
        "happiness",
        "neutral",
        "worry",
        "happiness",
        "neutral",
    ]
    assert set(result["sentiment"].unique()) <= {
        "anger",
        "happiness",
        "love",
        "neutral",
        "sadness",
        "surprise",
        "worry",
    }


def test_consolidate_sentiments_drops_unmapped_labels_without_mutating_input():
    df = make_frame(["worry", "admiration", "sadness"])

    result = consolidate_sentiments(df)

    # Unknown labels become NaN via Series.map() and must be dropped, not kept.
    assert result["sentiment"].tolist() == ["worry", "sadness"]
    # The caller's frame must be left untouched (df.copy() on entry).
    assert df["sentiment"].tolist() == ["worry", "admiration", "sadness"]


def test_clean_text_returns_empty_string_for_non_string_values():
    # CSV columns can contain NaN/floats; the isinstance guard keeps the
    # row-level pipeline from raising on them.
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
