"""
Data pipeline tests: corpus loading/splitting and the sliding-window
dataset that turns token ids into (input, target) pairs.
"""

import pytest
import torch

from data.corpus import load_corpus, train_val_split
from data.prepare_dialog import (build_corpus, clean_utterance, format_dialogue,
                                 parse_raw_dialogues)
from dataset import TextDataset, create_dataloader


class RangeTokenizer:
    """Fake tokenizer: text of length n -> token ids [0, 1, ..., n-1]."""

    def encode(self, text):
        return list(range(len(text)))


def test_corpus_loads():
    text = load_corpus()
    assert len(text) > 1_000_000
    assert text.startswith("First Citizen:")


def test_train_val_split_is_contiguous():
    text = "abcdefghij" * 10
    train, val = train_val_split(text, val_ratio=0.1)
    assert train + val == text
    assert len(val) == 10


def test_train_val_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        train_val_split("abc", val_ratio=1.0)


def test_target_is_input_shifted_by_one():
    ds = TextDataset("x" * 20, RangeTokenizer(), context_length=4, stride=3)
    for X, Y in ds:
        assert X.shape == Y.shape == (4,)
        assert torch.equal(Y, X + 1)


def test_window_count_and_stride():
    ds = TextDataset("x" * 20, RangeTokenizer(), context_length=4, stride=3)
    # starts at 0, 3, 6, ... while start + context_length < 20
    assert len(ds) == len(range(0, 20 - 4, 3))
    assert ds[1][0][0].item() == 3


def test_too_short_text_raises():
    with pytest.raises(ValueError):
        TextDataset("abc", RangeTokenizer(), context_length=4, stride=1)


def test_dataloader_batch_shape():
    loader = create_dataloader("x" * 200, RangeTokenizer(),
                               context_length=8, stride=8, batch_size=4)
    X, Y = next(iter(loader))
    assert X.shape == Y.shape == (4, 8)
    assert X.dtype == torch.long


# ---------------------------------------------------------------------------
# DailyDialog chat corpus (data/prepare_dialog.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, clean", [
    ("Say , Jim , how about it ?", "Say, Jim, how about it?"),
    ("I ’ m sure it ’ s fine !", "I'm sure it's fine!"),
    ("I ' Ve got it", "I've got it"),
    ("It costs $ 5 ( or so ) ...", "It costs $5 (or so)..."),
    ("  extra   spaces  ", "extra spaces"),
])
def test_clean_utterance(raw, clean):
    assert clean_utterance(raw) == clean


def test_format_dialogue_alternates_user_and_bot():
    assert format_dialogue(["Hi .", "Hello .", "Bye ."]) == (
        "User: Hi .\nBot: Hello .\nUser: Bye ."
    )


def test_parse_raw_dialogues_splits_turns_and_skips_single_turns():
    raw = "Hi , there . __eou__ Hello ! __eou__\nOnly one turn . __eou__\n"
    assert parse_raw_dialogues(raw) == [["Hi, there.", "Hello!"]]


def test_build_corpus_separates_and_dedupes_conversations():
    a, b = ["Hi.", "Hello."], ["Bye.", "See you."]
    corpus, duplicates = build_corpus([a, b, a])
    assert duplicates == 1
    assert corpus == (
        "User: Hi.\nBot: Hello.\n\n"
        "User: Bye.\nBot: See you.\n"
    )
