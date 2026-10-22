"""
Training-infrastructure tests: the learning-rate schedule, checkpoint
save/load, and train.py's tokenizer reuse.
"""

import math

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


def test_bits_per_char():
    # ln 2 nats per token, one character per token = exactly 1 bit/char.
    assert train.bits_per_char(math.log(2), 1.0) == pytest.approx(1.0)
    # Same loss per token over tokens twice as long = half the bits per char.
    assert train.bits_per_char(math.log(2), 2.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def make_tokenizer(num_merges=20):
    tok = BPETokenizer()
    tok.train(TEXT, num_merges=num_merges)
    return tok


@pytest.mark.parametrize("tie_weights", [True, False])
def test_checkpoint_round_trip(tmp_path, tie_weights):
    tok = make_tokenizer()
    model_config = {"vocab_size": tok.vocab_size, "embedding_dim": 32,
                    "num_heads": 4, "num_layers": 2, "max_seq_length": 16,
                    "dropout": 0.1, "tie_weights": tie_weights}
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
    assert (loaded.lm_head.weight is loaded.token_embedding.weight) == tie_weights


def test_checkpoint_from_before_weight_tying_loads_untied(tmp_path):
    # Older checkpoints have no "tie_weights" key and two separate
    # matrices; they must not be loaded into a tied model.
    tok = make_tokenizer()
    model_config = {"vocab_size": tok.vocab_size, "embedding_dim": 32,
                    "num_heads": 4, "num_layers": 2, "max_seq_length": 16,
                    "dropout": 0.1}
    model = MiniLLM(**model_config, tie_weights=False).eval()

    save_checkpoint(model, model_config, tok, epoch=1, val_loss=2.0,
                    checkpoint_dir=tmp_path)
    loaded, _, _ = load_checkpoint(tmp_path)

    assert loaded.lm_head.weight is not loaded.token_embedding.weight
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


def test_tokenizer_reuse_reads_given_checkpoint_dir(tmp_path, monkeypatch):
    # A tokenizer saved in a custom folder is found there, not in the default.
    make_tokenizer(num_merges=20).save(tmp_path / config.TOKENIZER_FILENAME)
    monkeypatch.setattr(config, "NUM_MERGES", 20)

    def fail(*args, **kwargs):
        raise AssertionError("tokenizer should have been reused, not retrained")
    monkeypatch.setattr(BPETokenizer, "train", fail)

    tok = train.load_or_train_tokenizer(TEXT, tmp_path)
    assert len(tok.merges) == 20


# ---------------------------------------------------------------------------
# Command-line options in train.py
# ---------------------------------------------------------------------------

def test_default_args_use_config():
    args = train.parse_args([])
    assert args.corpus is None
    assert args.checkpoint_dir is None
    assert args.epochs == config.EPOCHS
    assert args.sample_prompt == config.SAMPLE_PROMPT


def test_epochs_option():
    assert train.parse_args(["--epochs", "3"]).epochs == 3


def test_sample_prompt_turns_backslash_n_into_newline():
    args = train.parse_args(["--sample-prompt", "User: hi\nBot:"])
    assert args.sample_prompt == "User: hi\nBot:"


def test_default_corpus_saves_to_default_checkpoint_dir():
    assert train.resolve_checkpoint_dir(None, None) == train.default_checkpoint_dir()


def test_custom_corpus_gets_its_own_checkpoint_dir():
    # Training on a new corpus must not overwrite the default model.
    path = train.resolve_checkpoint_dir("data/dialog_corpus.txt", None)
    assert path == train.default_checkpoint_dir() / "dialog_corpus"


def test_explicit_checkpoint_dir_wins(tmp_path):
    assert train.resolve_checkpoint_dir("data/dialog_corpus.txt", tmp_path) == tmp_path


def test_refuses_to_overwrite_a_trained_model(tmp_path):
    (tmp_path / config.MODEL_CHECKPOINT_FILENAME).write_bytes(b"trained")
    with pytest.raises(SystemExit):
        train.check_not_overwriting(tmp_path, overwrite=False)
    train.check_not_overwriting(tmp_path, overwrite=True)  # explicitly allowed


def test_empty_checkpoint_dir_is_fine_to_train_into(tmp_path):
    train.check_not_overwriting(tmp_path / "new_run", overwrite=False)
