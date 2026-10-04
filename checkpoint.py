"""
checkpoint.py
=============

Saving and loading a trained MiniLLM together with its tokenizer.

Why this exists:
    Before this, train.py built a model, trained it, and threw it away
    when the script exited. Anything that wanted to *use* the model
    (generate.py) would have had to retrain from scratch every time.

A checkpoint is two files in config.CHECKPOINT_DIR:

    mini_llm.pt      model weights + the hyperparameters needed to
                     rebuild an identically shaped MiniLLM
    tokenizer.json   the BPE tokenizer the model was trained with

Both are needed: the model's token ids only mean something with the
exact vocabulary and merges that produced them.

The hyperparameters are stored inside the checkpoint (rather than read
from config.py at load time) so that changing config.py later -- e.g.
a bigger EMBEDDING_DIM for the next experiment -- doesn't break loading
an older checkpoint.
"""

from pathlib import Path

import torch

import config
from bpe_tokenizer import BPETokenizer
from model import MiniLLM


def default_checkpoint_dir() -> Path:
    """config.CHECKPOINT_DIR resolved against the project root, not the cwd."""
    return Path(__file__).resolve().parent / config.CHECKPOINT_DIR


def save_checkpoint(model, model_config, tokenizer, epoch, val_loss,
                    checkpoint_dir=None):
    """
    Write model weights + hyperparameters and the tokenizer to
    checkpoint_dir (defaults to config.CHECKPOINT_DIR).

    model_config: the keyword arguments MiniLLM was constructed with,
                  e.g. {"vocab_size": 569, "embedding_dim": 128, ...}
    """
    checkpoint_dir = Path(checkpoint_dir or default_checkpoint_dir())
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        checkpoint_dir / config.MODEL_CHECKPOINT_FILENAME,
    )
    tokenizer.save(checkpoint_dir / config.TOKENIZER_FILENAME)


def load_checkpoint(checkpoint_dir=None, map_location="cpu"):
    """
    Rebuild the model and tokenizer saved by save_checkpoint().

    Returns (model, tokenizer, info) where info holds the epoch and
    val_loss the checkpoint was saved at. The model is returned on
    `map_location` and in eval mode, ready for generation.
    """
    checkpoint_dir = Path(checkpoint_dir or default_checkpoint_dir())
    model_path = checkpoint_dir / config.MODEL_CHECKPOINT_FILENAME
    tokenizer_path = checkpoint_dir / config.TOKENIZER_FILENAME

    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            f"Run `python train.py` first to train and save a model."
        )

    checkpoint = torch.load(model_path, map_location=map_location)

    model = MiniLLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()

    tokenizer = BPETokenizer.load(tokenizer_path)

    info = {"epoch": checkpoint["epoch"], "val_loss": checkpoint["val_loss"]}
    return model, tokenizer, info
