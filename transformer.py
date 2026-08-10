"""
transformer.py
==============

Transformer Block for the Mini LLM — the next stage after attention.py.

Where this fits in the pipeline:

    Combined Embeddings (token + position)
                |
           Self-Attention          <-- attention.py
                |
        Transformer Block          <-- this file
                |
    Remaining Transformer Layers (stacked blocks)
                |
    Final LayerNorm + Vocabulary Projection  (future gpt.py)

This file does NOT reimplement attention — it imports MultiHeadAttention
directly from attention.py and wires it into a full Pre-Norm block:

    X1 = X  + Dropout(MHA(LayerNorm(X)))
    X2 = X1 + FFN(LayerNorm(X1))

Two new pieces are added here that attention.py doesn't have on its own:

    1. FeedForward   - the per-token Linear -> GELU -> Linear
    2. Dropout       - applied after attention and inside the FFN,
                        using config.DROPOUT (attention.py itself has
                        no dropout, so it's added at the block level)

Also builds TransformerBlocks, a stack of `config.NUM_LAYERS` blocks —
the piece that will sit between embeddings and the final vocab head.

Run this file directly to see shapes at every stage:

    python transformer.py
"""

import torch
import torch.nn as nn

import config
from attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# 1. Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: Linear -> GELU -> Linear.

    Where MultiHeadAttention lets tokens borrow information from each
    other, FeedForward processes each token's (now context-aware)
    representation independently and nonlinearly. Every position goes
    through the exact same two linear layers, applied one token at a
    time (no mixing across the sequence dimension).

    The hidden layer is widened by hidden_mult (4x, matching GPT-style
    models) before being projected back down to embedding_dim.
    """

    def __init__(self, embedding_dim, hidden_mult=4, dropout=0.0):
        super().__init__()
        hidden_dim = embedding_dim * hidden_mult

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, embedding_dim)
        Returns: (batch, seq_len, embedding_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    One Pre-Norm Transformer block, combining attention.py's
    MultiHeadAttention with a FeedForward network:

        X1 = X  + Dropout(MHA(LayerNorm(X)))
        X2 = X1 + FFN(LayerNorm(X1))

    Pre-Norm means LayerNorm is applied *before* each sub-layer, and the
    sub-layer's output is added onto the original (un-normalized)
    residual stream. This keeps the residual path clean, which makes
    stacks of many blocks much easier to train than Post-Norm.

    attention.py's MultiHeadAttention.forward returns (out, attn_maps) —
    this block passes attn_maps through so you can still inspect what
    each head is attending to, e.g. for debugging or visualization.
    """

    def __init__(self, embedding_dim=None, num_heads=None,
                 max_seq_length=None, dropout=None):
        super().__init__()
        embedding_dim = embedding_dim or config.EMBEDDING_DIM
        num_heads = num_heads or config.NUM_HEADS
        max_seq_length = max_seq_length or config.MAX_SEQ_LENGTH
        dropout = config.DROPOUT if dropout is None else dropout

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, max_seq_length)
        self.attn_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, dropout=dropout)

    def forward(self, x, return_attn=False):
        """
        x: (batch, seq_len, embedding_dim)
        Returns: (batch, seq_len, embedding_dim), same shape as input
                 (plus attn_maps if return_attn=True)
        """
        attn_out, attn_maps = self.attn(self.ln1(x))
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ln2(x))

        if return_attn:
            return x, attn_maps
        return x


# ---------------------------------------------------------------------------
# 3. Stack of Transformer Blocks
# ---------------------------------------------------------------------------
class TransformerBlocks(nn.Module):
    """
    A stack of `num_layers` TransformerBlock instances, applied in
    sequence. Each block refines the representation a bit further —
    this is the piece that will sit between your embeddings and the
    final LayerNorm + vocabulary projection in the full GPT model.
    """

    def __init__(self, num_layers=None, embedding_dim=None, num_heads=None,
                 max_seq_length=None, dropout=None):
        super().__init__()
        num_layers = num_layers or config.NUM_LAYERS

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, max_seq_length, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: (batch, seq_len, embedding_dim)
        Returns: (batch, seq_len, embedding_dim)
        """
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Same dog/bites/man example used in attention.py's demo, so shapes
    # can be compared directly against that file's output.
    vocab = {"dog": 0, "bites": 1, "man": 2}
    token_ids = torch.tensor([[vocab["dog"], vocab["bites"], vocab["man"]]])  # (1, 3)
    B, T = token_ids.shape

    token_embedding = nn.Embedding(len(vocab), config.EMBEDDING_DIM)
    position_embedding = nn.Embedding(config.MAX_SEQ_LENGTH, config.EMBEDDING_DIM)

    position_ids = torch.arange(T).unsqueeze(0)
    x = token_embedding(token_ids) + position_embedding(position_ids)  # (1, 3, EMBEDDING_DIM)

    print("Config: EMBEDDING_DIM =", config.EMBEDDING_DIM,
          "| NUM_HEADS =", config.NUM_HEADS,
          "| NUM_LAYERS =", config.NUM_LAYERS,
          "| MAX_SEQ_LENGTH =", config.MAX_SEQ_LENGTH,
          "| DROPOUT =", config.DROPOUT)
    print("Input shape (combined token+position embeddings):", x.shape)

    print("\n--- 1. Single Transformer block ---")
    block = TransformerBlock()
    out, attn_maps = block(x, return_attn=True)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "block must preserve shape"
    print("Number of attention maps (one per head):", len(attn_maps))

    print(f"\n--- 2. Stack of {config.NUM_LAYERS} Transformer blocks ---")
    blocks = TransformerBlocks()
    out = blocks(x)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "stack must preserve shape"

    print("\nOK — shape preserved through one block and the full stack.")
