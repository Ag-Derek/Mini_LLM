"""
Tokenizer tests: BPE (the one the model trains with) plus the simpler
character and word tokenizers it grew out of.
"""

import config
from bpe_tokenizer import BPETokenizer
from tokenizer import CharTokenizer
from word_tokenizer import WordTokenizer


SAMPLE = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
    (indented line)  with  double  spaces
"""


def trained_bpe(num_merges=50):
    tok = BPETokenizer()
    tok.train(SAMPLE, num_merges=num_merges)
    return tok


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------

def test_bpe_exact_round_trip_keeps_whitespace():
    tok = trained_bpe()
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_bpe_special_tokens_come_first():
    tok = trained_bpe()
    for i, special in enumerate(config.SPECIAL_TOKENS):
        assert tok.stoi[special] == i


def test_bpe_merges_shorten_the_sequence():
    tok = trained_bpe()
    # Without merges every character would be its own token.
    assert len(tok.encode(SAMPLE)) < len(SAMPLE)
    assert len(tok.merges) > 0


def test_bpe_unseen_word_from_known_characters_has_no_unk():
    tok = trained_bpe()
    unk = tok.stoi[config.UNK_TOKEN]
    ids = tok.encode(" peaches")  # never in SAMPLE, but its letters are
    assert unk not in ids
    assert tok.decode(ids) == " peaches"


def test_bpe_unseen_character_maps_to_unk():
    tok = trained_bpe()
    unk = tok.stoi[config.UNK_TOKEN]
    assert tok.encode("é") == [unk]


def test_bpe_save_load_round_trip(tmp_path):
    tok = trained_bpe()
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.stoi == tok.stoi
    assert loaded.merges == tok.merges
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)


def test_bpe_encode_cache_does_not_change_results():
    tok = trained_bpe()
    first = tok.encode(SAMPLE)
    second = tok.encode(SAMPLE)  # served from the per-chunk cache
    assert first == second


# ---------------------------------------------------------------------------
# Character and word tokenizers
# ---------------------------------------------------------------------------

def test_char_tokenizer_round_trip_is_lowercased():
    tok = CharTokenizer()
    tok.train(SAMPLE)
    assert tok.decode(tok.encode("Speak, speak.")) == "speak, speak."


def test_word_tokenizer_token_level_round_trip():
    # WordTokenizer normalizes whitespace, so only re-encoding the decoded
    # text is guaranteed to give the same ids back.
    tok = WordTokenizer()
    tok.train(SAMPLE)
    ids = tok.encode("Speak, speak.")
    assert tok.encode(tok.decode(ids)) == ids
