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
    tokenizer.json   the tokenizer the model was trained with: our own
                     BPE (bpe_tokenizer.py), or for a model converted
                     from GPT-2 (load_gpt2.py) just {"type": "tiktoken"}

Both are needed: the model's token ids only mean something with the
exact vocabulary and merges that produced them.

The hyperparameters are stored inside the checkpoint (rather than read
from config.py at load time) so that changing config.py later -- e.g.
a bigger EMBEDDING_DIM for the next experiment -- doesn't break loading
an older checkpoint.
"""

import json
import re
from pathlib import Path

import torch

import config
from bpe_tokenizer import BPETokenizer
from model import MiniLLM

# Matches the per-head attention weights checkpoints had before
# CausalSelfAttention, e.g. "transformer.blocks.0.attn.heads.3.query.weight".
_PER_HEAD_KEY = re.compile(r"(transformer\.blocks\.\d+\.attn\.)heads\.(\d+)\.(.+)")


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


def load_tokenizer(path):
    """
    Load whichever tokenizer a checkpoint was saved with. GPT2Tokenizer
    writes a "type" field; files without one are our own BPETokenizer.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") == "tiktoken":
        from gpt2_tokenizer import GPT2Tokenizer  # needs tiktoken; only import when used
        return GPT2Tokenizer.load(path)
    return BPETokenizer.load(path)


def upgrade_per_head_attention(state):
    """
    Convert a state dict saved with the old per-head attention
    (attention.py's MultiHeadAttention: one query/key/value Linear per
    head, plus a causal_mask buffer each) to CausalSelfAttention's single
    qkv Linear. Returns a new state dict; state dicts that are already in
    the new layout come back unchanged.

    The fused qkv weight is the per-head weights stacked in the order
    CausalSelfAttention splits them: all heads' queries, then keys, then
    values, each in head order. The old Linears had no bias, so the qkv
    bias is zero -- the converted model computes exactly the same thing.
    """
    per_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    upgraded = {}
    for key, tensor in state.items():
        match = _PER_HEAD_KEY.fullmatch(key)
        if match is None:
            upgraded[key] = tensor
            continue
        prefix, head, name = match[1], int(match[2]), match[3]
        if name == "causal_mask":
            continue  # the fused attention builds its mask on the fly
        name = name.removesuffix(".weight")
        per_layer.setdefault(prefix, {}).setdefault(name, {})[head] = tensor

    for prefix, parts in per_layer.items():
        stacked = [torch.cat([parts[name][h] for h in sorted(parts[name])])
                   for name in ("query", "key", "value")]
        upgraded[prefix + "qkv.weight"] = torch.cat(stacked)
        upgraded[prefix + "qkv.bias"] = torch.zeros(upgraded[prefix + "qkv.weight"].shape[0])
    return upgraded


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

    # Checkpoints saved before weight tying existed have no "tie_weights"
    # key and separate embedding / lm_head weights. Loading those into a
    # tied model would silently let one matrix overwrite the other.
    # Likewise, checkpoints from before the GPT-2-compatible GELU option
    # were trained with the exact GELU.
    model_config = {"tie_weights": False, "gelu_approximate": "none",
                    **checkpoint["model_config"]}
    model = MiniLLM(**model_config)
    model.load_state_dict(upgrade_per_head_attention(checkpoint["model_state"]))
    model.to(map_location)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_path)

    info = {"epoch": checkpoint["epoch"], "val_loss": checkpoint["val_loss"]}
    return model, tokenizer, info
