"""
Training-infrastructure tests: the learning-rate schedule, checkpoint
save/load, and train.py's tokenizer reuse.
"""

import pytest
import torch

import config
import train
from bpe_tokenizer import BPETokenizer
from checkpoint import load_checkpoint, save_checkpoint
from model import MiniLLM

TEXT = "To be, or not to be, that is the question:\n" * 20


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def test_lr_warmup_then_cosine_decay():
    total = 1000
    lrs = [train.get_lr(step, total) for step in range(total + 1)]
    warmup = config.WARMUP_STEPS

    # Linear warmup up to the peak...
    assert lrs[0] == pytest.approx(config.LEARNING_RATE / warmup)
    assert lrs[warmup - 1] == pytest.approx(config.LEARNING_RATE)
    assert all(a < b for a, b in zip(lrs[:warmup - 1], lrs[1:warmup]))
    # ...then never increases again, ending at MIN_LR.
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))
    assert lrs[total] == pytest.approx(config.MIN_LR)
    assert min(lrs[warmup:]) >= config.MIN_LR - 1e-12


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def make_tokenizer(num_merges=20):
    tok = BPETokenizer()
    tok.train(TEXT, num_merges=num_merges)
    return tok


def test_checkpoint_round_trip(tmp_path):
    tok = make_tokenizer()
    model_config = {"vocab_size": tok.vocab_size, "embedding_dim": 32,
                    "num_heads": 4, "num_layers": 2, "max_seq_length": 16,
                    "dropout": 0.1}
    model = MiniLLM(**model_config).eval()

    save_checkpoint(model, model_config, tok, epoch=3, val_loss=1.25,
                    checkpoint_dir=tmp_path)
    loaded, loaded_tok, info = load_checkpoint(tmp_path)

    assert info == {"epoch": 3, "val_loss": 1.25}
    assert not loaded.training  # returned ready for generation
    assert loaded_tok.encode(TEXT) == tok.encode(TEXT)

    ids = torch.tensor([tok.encode(TEXT)[:16]])
    with torch.no_grad():
        assert torch.equal(loaded(ids), model(ids))


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nothing-here")


# ---------------------------------------------------------------------------
# Tokenizer reuse in train.py
# ---------------------------------------------------------------------------

@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    # An absolute CHECKPOINT_DIR overrides the project-root-relative default.
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path))
    return tmp_path


def test_matching_saved_tokenizer_is_reused(checkpoint_dir, monkeypatch):
    make_tokenizer(num_merges=20).save(checkpoint_dir / config.TOKENIZER_FILENAME)
    monkeypatch.setattr(config, "NUM_MERGES", 20)

    def fail(*args, **kwargs):
        raise AssertionError("tokenizer should have been reused, not retrained")
    monkeypatch.setattr(BPETokenizer, "train", fail)

    tok = train.load_or_train_tokenizer(TEXT)
    assert len(tok.merges) == 20


def test_mismatched_saved_tokenizer_is_retrained(checkpoint_dir, monkeypatch):
    make_tokenizer(num_merges=20).save(checkpoint_dir / config.TOKENIZER_FILENAME)
    monkeypatch.setattr(config, "NUM_MERGES", 10)

    tok = train.load_or_train_tokenizer(TEXT)
    assert len(tok.merges) == 10


def test_no_saved_tokenizer_trains_one(checkpoint_dir, monkeypatch):
    monkeypatch.setattr(config, "NUM_MERGES", 15)
    tok = train.load_or_train_tokenizer(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT
