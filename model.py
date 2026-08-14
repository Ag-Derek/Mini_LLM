"""
Mini LLM — full model, wiring embeddings + TransformerBlocks + output head.

Where this fits in the pipeline:

    BPE Token IDs
          |
    Token + Position Embeddings
          |
    TransformerBlocks (stack of TransformerBlock)   <-- transformer.py
          |
    Final LayerNorm
          |
    Vocabulary Projection (lm_head)
          |
        logits

This file does not reimplement attention or the transformer block — it
imports TransformerBlocks from transformer.py and wraps it with the
embedding layers on one side and the vocab head on the other, giving a
single end-to-end nn.Module you can call as model(token_ids).

Run this file directly to see shapes at every stage:

    python model.py
"""

import torch
import torch.nn as nn

import config
from transformer import TransformerBlocks


class MiniLLM(nn.Module):
    """
    End-to-end GPT-style language model.

    forward(token_ids) -> logits over the vocabulary at every position,
    ready to be compared against target token ids with CrossEntropyLoss.
    """

    def __init__(self, vocab_size, embedding_dim=None, num_heads=None,
                 num_layers=None, max_seq_length=None, dropout=None):
        super().__init__()
        embedding_dim = embedding_dim or config.EMBEDDING_DIM
        max_seq_length = max_seq_length or config.MAX_SEQ_LENGTH
        dropout = config.DROPOUT if dropout is None else dropout

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.transformer = TransformerBlocks(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) of integer token ids
        Returns: logits, shape (batch, seq_len, vocab_size)
        """
        B, T = token_ids.shape

        position_ids = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        x = self.embed_dropout(x)

        x = self.transformer(x)

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 568  # matches your trained BPE tokenizer
    model = MiniLLM(vocab_size=vocab_size)

    token_ids = torch.tensor([[1, 8]])  # (1, 2)
    print("Input shape:", token_ids.shape)

    logits = model(token_ids)
    print("Output shape:", logits.shape)
    assert logits.shape == (token_ids.shape[0], token_ids.shape[1], vocab_size)

    # also check the (1, 8) case explicitly, matching the plan
    token_ids_8 = torch.randint(0, vocab_size, (1, 8))
    logits_8 = model(token_ids_8)
    print("Input shape:", token_ids_8.shape, "-> Output shape:", logits_8.shape)
    assert logits_8.shape == (1, 8, vocab_size)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    print("\nOK — forward pass connected end to end.")
