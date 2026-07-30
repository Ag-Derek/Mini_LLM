"""
attention.py
============

Self-Attention for the Mini LLM project.

Where this fits in the pipeline:

    Raw Text -> Tokenizer -> Token IDs -> Token Embedding
                                              |
                                    + Position Embedding
                                              |
                                     Combined Embeddings
                                              |
                                         Self-Attention   <-- this file
                                              |
                              Remaining Transformer Layers

By the time a sequence reaches this module, every vector already encodes
"what word is this" + "where is this word". Attention is the mechanism
that lets each position look at every other position and decide how much
to borrow from them when building its own updated representation.

This file builds attention in three stages, each one a class you can run
and inspect on its own:

    1. SelfAttentionHead     - a single scaled dot-product attention head
    2. CausalSelfAttentionHead - the same, but blocked from looking ahead
       (needed for a language model that predicts the next token)
    3. MultiHeadAttention    - several causal heads running in parallel

Run this file directly to see shapes and a worked numeric example at
every stage:

    python attention.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. A single scaled dot-product attention head
# ---------------------------------------------------------------------------
class SelfAttentionHead(nn.Module):
    """
    One attention head.

    Each token's combined embedding (token + position) is projected into
    three different vectors:

        Query (Q)  - "what am I looking for?"
        Key   (K)  - "what do I contain, for others to find me by?"
        Value (V)  - "what do I actually offer, once found?"

    Attention scores are computed as Q . K^T, scaled down, turned into a
    probability distribution with softmax, and used to take a weighted
    average of the Value vectors. This weighted average is the token's
    new, context-aware representation.
    """

    def __init__(self, embedding_dim, head_dim):
        super().__init__()
        self.head_dim = head_dim

        # Linear projections that produce Q, K, V from the input embeddings.
        # No bias, matching the convention used in GPT-style models.
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, embedding_dim) - combined token+position embeddings

        Returns: (batch, seq_len, head_dim)
        """
        B, T, _ = x.shape

        Q = self.query(x)  # (B, T, head_dim)
        K = self.key(x)    # (B, T, head_dim)
        V = self.value(x)  # (B, T, head_dim)

        # Raw attention scores: how much does each position's query match
        # every position's key?
        # (B, T, head_dim) @ (B, head_dim, T) -> (B, T, T)
        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(head_dim). Without this, dot products grow large as
        # head_dim grows, pushing softmax into extremely peaked (or flat)
        # distributions and making gradients unstable.
        scores = scores / math.sqrt(self.head_dim)

        # Turn scores into a probability distribution over positions,
        # per row (i.e. per query position).
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # Weighted sum of Value vectors using the attention distribution.
        out = attn_weights @ V  # (B, T, head_dim)

        return out, attn_weights


# ---------------------------------------------------------------------------
# 2. Causal (masked) self-attention
# ---------------------------------------------------------------------------
class CausalSelfAttentionHead(nn.Module):
    """
    Same as SelfAttentionHead, but a token is not allowed to attend to
    tokens that come after it.

    Why this matters:

        We are training a language model to predict the next token.
        If position 2 could freely attend to position 3, it could simply
        "peek" at the answer it's supposed to predict. Causal masking
        prevents this by setting the attention score for any (query, key)
        pair where key_position > query_position to -infinity, before
        the softmax. After softmax, -infinity becomes 0 probability.
    """

    def __init__(self, embedding_dim, head_dim, max_seq_length):
        super().__init__()
        self.head_dim = head_dim

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        # A lower-triangular matrix of 1s marks which (query, key) pairs
        # are allowed to see each other. Registered as a buffer (not a
        # learned parameter) so it moves with the module to GPU/CPU and
        # gets saved/loaded with the model, but is never updated by the
        # optimizer.
        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)

        # Only use the top-left (T, T) slice of the mask, in case the
        # current sequence is shorter than max_seq_length.
        mask = self.causal_mask[:T, :T]

        # Wherever mask == 0 (future positions), set the score to -inf
        # so softmax assigns them exactly 0 probability.
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V

        return out, attn_weights


# ---------------------------------------------------------------------------
# 3. Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Runs several CausalSelfAttentionHead instances in parallel, each with
    a smaller head_dim, then concatenates their outputs back together.

    Why multiple heads instead of one big head?

        A single attention head is forced to learn one notion of
        "relatedness" between tokens. Multiple smaller heads can each
        specialize: one might learn to track subject-verb agreement,
        another might track nearby words, another long-range references,
        and so on. Concatenating their outputs lets the next layer draw
        on all of these perspectives at once.

    The math is arranged so total compute stays comparable to one big
    head: if embedding_dim = 768 and num_heads = 12, each head works with
    head_dim = 64, and the 12 outputs of size 64 concatenate back to 768.
    """

    def __init__(self, embedding_dim, num_heads, max_seq_length):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(embedding_dim, head_dim, max_seq_length)
            for _ in range(num_heads)
        ])

        # Final linear layer to mix information across heads after
        # concatenation. Without this, the heads' outputs would just sit
        # side by side with no interaction.
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, embedding_dim)
        Returns: (batch, seq_len, embedding_dim)
        """
        head_outputs = []
        attn_maps = []
        for head in self.heads:
            out, attn_weights = head(x)
            head_outputs.append(out)
            attn_maps.append(attn_weights)

        # Concatenate along the last dimension: (B, T, head_dim * num_heads)
        # which equals (B, T, embedding_dim).
        combined = torch.cat(head_outputs, dim=-1)

        out = self.proj(combined)

        return out, attn_maps


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Reuse the dog/bites/man example from the positional embeddings step.
    vocab = {"dog": 0, "bites": 1, "man": 2}
    embedding_dim = 8
    max_seq_length = 8

    token_ids = torch.tensor([[vocab["dog"], vocab["bites"], vocab["man"]]])  # (1, 3)
    B, T = token_ids.shape

    token_embedding = nn.Embedding(len(vocab), embedding_dim)
    position_embedding = nn.Embedding(max_seq_length, embedding_dim)

    position_ids = torch.arange(T).unsqueeze(0)  # (1, 3) -> [[0, 1, 2]]
    x = token_embedding(token_ids) + position_embedding(position_ids)  # (1, 3, 8)

    print("Input shape (combined token+position embeddings):", x.shape)

    print("\n--- 1. Single head, no causal mask ---")
    head = SelfAttentionHead(embedding_dim, head_dim=8)
    out, weights = head(x)
    print("Output shape:", out.shape)
    print("Attention weights (each row sums to 1):\n", weights[0])

    print("\n--- 2. Single head, causal mask ---")
    causal_head = CausalSelfAttentionHead(embedding_dim, head_dim=8, max_seq_length=max_seq_length)
    out, weights = causal_head(x)
    print("Attention weights (upper triangle should be 0, i.e. no peeking ahead):")
    print(weights[0])

    print("\n--- 3. Multi-head attention (4 heads) ---")
    mha = MultiHeadAttention(embedding_dim, num_heads=4, max_seq_length=max_seq_length)
    out, attn_maps = mha(x)
    print("Output shape:", out.shape)
    print("Number of attention maps (one per head):", len(attn_maps))
    print("First head's attention weights:\n", attn_maps[0][0])
