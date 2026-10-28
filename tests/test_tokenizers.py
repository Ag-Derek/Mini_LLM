"""
Tokenizer tests: BPE (the one the model trains with) plus the simpler
character and word tokenizers it grew out of.
"""

from collections import Counter

import pytest

import config
from bpe_tokenizer import BPETokenizer
from checkpoint import load_tokenizer
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


def reference_merges(text, num_merges):
    """
    BPE training done the simple, slow way: recount every pair in the
    corpus before each merge. BPETokenizer.train() keeps running counts
    instead; both must learn the same merges in the same order.
    """
    word_freqs = Counter(BPETokenizer._CHUNK_PATTERN.findall(text))
    splits = {word: list(word) for word in word_freqs}
    merges = []
    for _ in range(num_merges):
        pair_freq = Counter()
        for word, freq in word_freqs.items():
            for pair in zip(splits[word], splits[word][1:]):
                pair_freq[pair] += freq
        if not pair_freq:
            break
        # Most frequent pair; ties go to the alphabetically first one.
        best = min(pair_freq, key=lambda pair: (-pair_freq[pair], pair))
        merges.append(best)
        for word in splits:
            splits[word] = BPETokenizer._merge_pair(splits[word], best, best[0] + best[1])
    return merges


def test_bpe_training_matches_simple_reference():
    # Enough merges to run out of pairs, which also covers early stopping
    # and repeated symbols like the double spaces in SAMPLE.
    tok = trained_bpe(num_merges=1000)
    learned = sorted(tok.merges, key=tok.merges.get)
    assert learned == reference_merges(SAMPLE, 1000)
    assert len(learned) < 1000  # stopped early: every chunk fully merged


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


# ---------------------------------------------------------------------------
# GPT-2 tokenizer (pretrained; tiktoken downloads its merge table once)
# ---------------------------------------------------------------------------

def test_gpt2_tokenizer_round_trips_any_text():
    pytest.importorskip("tiktoken")
    from gpt2_tokenizer import END_OF_TEXT, GPT2Tokenizer

    tok = GPT2Tokenizer()
    assert tok.vocab_size == 50257
    text = SAMPLE + "café 🙂 \t\n"  # byte-level: nothing becomes <unk>
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode("Hi" + END_OF_TEXT) == tok.encode("Hi") + [tok.eot_id]


def test_load_tokenizer_picks_the_saved_type(tmp_path):
    pytest.importorskip("tiktoken")
    from gpt2_tokenizer import GPT2Tokenizer

    gpt2_path, bpe_path = tmp_path / "gpt2.json", tmp_path / "bpe.json"
    GPT2Tokenizer().save(gpt2_path)
    trained_bpe().save(bpe_path)

    assert isinstance(load_tokenizer(gpt2_path), GPT2Tokenizer)
    assert isinstance(load_tokenizer(bpe_path), BPETokenizer)
