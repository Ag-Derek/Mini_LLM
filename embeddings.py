"""
embeddings.py
--------------
Step: Embeddings for the Mini LLM project.

Goal: understand what nn.Embedding actually does by building our own
version from scratch, then comparing it against PyTorch's built-in
implementation.

Key idea:
    An embedding layer is a LOOKUP TABLE.
    - Shape: (vocab_size, embedding_dim)
    - Given a token id, we return that row of the table.
    - The table itself is a trainable parameter (nn.Parameter), so
      gradients flow into it during backprop, just like any other
      weight matrix.

Mathematically:
    embedding(id) == one_hot(id) @ W

But we never actually build the one-hot vector — we just index
directly into W. That's the "lookup table" shortcut.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Our own embedding layer, built from scratch
# ---------------------------------------------------------------------------
class MyEmbedding(nn.Module):
    """
    A minimal re-implementation of nn.Embedding.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary. This determines
        how many rows the embedding table has.
    embedding_dim : int
        Size of each token's vector. This determines how many
        columns the embedding table has.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # The embedding table itself: one row per token, one column
        # per dimension. This is the ONLY thing this layer learns.
        #
        # We initialize with small random values (like nn.Embedding
        # does with a normal distribution) rather than zeros, so that
        # different tokens start out distinguishable from one another
        # and gradients have something to push against.
        self.weight = nn.Parameter(
        torch.empty(vocab_size, embedding_dim)
        )

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embedding vector for each token id.

        token_ids: LongTensor of shape (...,) — any shape, typically
                   (batch_size, sequence_length).

        returns:   FloatTensor of shape (..., embedding_dim)
        """
        # This is the entire "forward pass" of an embedding layer:
        # index into the table with the token ids. PyTorch's fancy
        # indexing handles arbitrary input shapes (1D, 2D, batched, etc.)
        # and — because self.weight is an nn.Parameter — autograd
        # automatically knows how to route gradients back into the
        # exact rows that were used.
        return self.weight[token_ids]


# ---------------------------------------------------------------------------
# 2. Sanity check: does our layer behave identically to nn.Embedding?
# ---------------------------------------------------------------------------
def compare_with_pytorch():
    torch.manual_seed(0)

    vocab_size = 10
    embedding_dim = 4

    my_emb = MyEmbedding(vocab_size, embedding_dim)

    # Build PyTorch's version and copy our weights into it, so we're
    # comparing the same underlying table (otherwise both would just
    # be randomly initialized differently and we couldn't compare).
    torch_emb = nn.Embedding(vocab_size, embedding_dim)
    with torch.no_grad():
        torch_emb.weight.copy_(my_emb.weight)

    print("\nEmbedding table:")
    print(my_emb.weight)

    token_ids = torch.tensor([[1, 4, 7], [2, 2, 9]])  # (batch=2, seq_len=3)

    print("\nLookup examples:")
    print("Token 1 ->", my_emb.weight[1])
    print("Token 4 ->", my_emb.weight[4])
    print("Token 7 ->", my_emb.weight[7])

    my_out = my_emb(token_ids)
    print("\nVerifying lookups...")

    print("weight[1] == output[0,0] :",
      torch.equal(my_emb.weight[1], my_out[0, 0]))

    print("weight[4] == output[0,1] :",
      torch.equal(my_emb.weight[4], my_out[0, 1]))

    print("weight[7] == output[0,2] :",
      torch.equal(my_emb.weight[7], my_out[0, 2]))

    torch_out = torch_emb(token_ids)

    print("Input token ids:\n", token_ids)
    print("\nMyEmbedding output shape:", my_out.shape)
    print("nn.Embedding output shape:", torch_out.shape)

    identical = torch.allclose(my_out, torch_out)
    print("\nOutputs identical:", identical)

    # ---- Demonstrate learning: gradients only touch used rows ----
    print("\n--- Gradient check ---")

    my_emb.zero_grad()

    gradient_test_ids = torch.tensor([1, 1, 3])

    out = my_emb(gradient_test_ids)

    loss = out.sum()

    loss.backward()

    grad = my_emb.weight.grad

    print("Gradient shape:", grad.shape)

    used_rows = (
    (grad.abs().sum(dim=1) > 0)
    .nonzero()
    .squeeze(-1)
    .tolist()
    )

    print("Rows with non-zero gradient:", used_rows)

    print("\nToken 1 was used twice, so its gradient accumulates:")
    print("grad row 1:", grad[1])

    print("\nToken 3 was used once:")
    print("grad row 3:", grad[3])

    print("\nToken 0 was unused, so it should be zero:")
    print("grad row 0:", grad[0])

    print("\nSummary")
    print("-"*40)


    print("\nEmbedding output:")
    print(my_out)
if __name__ == "__main__":
    compare_with_pytorch()
