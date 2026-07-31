"""
config.py
=========

Single source of truth for every hyperparameter and path used across the
Mini_LLM project.

Why this file exists:
    Before this, values like context_length, num_merges, and embedding_dim
    were hard-coded separately inside bpe_tokenizer.py, dataset.py,
    attention.py, and embeddings.py (often with different numbers in each
    file's __main__ demo block). That made it impossible to change one
    setting and trust it everywhere.

    Every module should now do:

        import config

    and read values from here (e.g. config.CONTEXT_LENGTH) instead of
    defining its own local constant.

Sections are grouped by pipeline stage, in the same order data flows
through the project: tokenizer -> dataset -> embeddings -> attention ->
(future) transformer -> training.
"""

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
# Shared across tokenizer.py / word_tokenizer.py / bpe_tokenizer.py so the
# same reserved ids mean the same thing regardless of which tokenizer is
# active.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# ---------------------------------------------------------------------------
# Tokenizer (bpe_tokenizer.py)
# ---------------------------------------------------------------------------
# Number of BPE merge operations to learn during training. Was 40/100/200
# across different __main__ demos before; one real corpus needs one real
# number. 500 is a reasonable starting point for a small corpus like Tiny
# Shakespeare -- raise it once vocab coverage is checked.
NUM_MERGES = 500

# ---------------------------------------------------------------------------
# Dataset (dataset.py)
# ---------------------------------------------------------------------------
# How many tokens the model sees per training example.
CONTEXT_LENGTH = 8

# How far the sliding window moves each step. stride < context_length
# means overlapping windows (more training samples from the same text);
# stride == context_length means no overlap.
STRIDE = 4

# Samples per training batch.
BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Embeddings (embeddings.py)
# ---------------------------------------------------------------------------
# Size of each token's embedding vector. attention.py's demo used 8 (to
# keep printed tensors readable); a real model needs a wider embedding.
EMBEDDING_DIM = 128

# ---------------------------------------------------------------------------
# Attention / Transformer (attention.py, future transformer.py)
# ---------------------------------------------------------------------------
# Longest sequence the model can ever be asked to attend over. This sizes
# the causal mask buffer in CausalSelfAttentionHead, so it must be >=
# CONTEXT_LENGTH. Keeping it equal to CONTEXT_LENGTH for now since we
# don't yet do generation past the training window.
MAX_SEQ_LENGTH = CONTEXT_LENGTH

# Number of parallel attention heads. EMBEDDING_DIM must be divisible by
# this (asserted in MultiHeadAttention).
NUM_HEADS = 4

# Number of stacked Transformer blocks (used once transformer.py exists).
NUM_LAYERS = 6

# Dropout probability (used once transformer.py / training add dropout
# layers).
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training (future train.py)
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
EPOCHS = 10

# ---------------------------------------------------------------------------
# Data paths (future data/corpus.py)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
CORPUS_FILENAME = "tiny_corpus.txt"


if __name__ == "__main__":
    # Quick sanity check: run `python config.py` to print every setting
    # and confirm nothing is missing or inconsistent.
    assert EMBEDDING_DIM % NUM_HEADS == 0, (
        "EMBEDDING_DIM must be divisible by NUM_HEADS"
    )
    assert MAX_SEQ_LENGTH >= CONTEXT_LENGTH, (
        "MAX_SEQ_LENGTH must be >= CONTEXT_LENGTH"
    )

    print("Special tokens:", SPECIAL_TOKENS)
    print("NUM_MERGES:", NUM_MERGES)
    print("CONTEXT_LENGTH:", CONTEXT_LENGTH)
    print("STRIDE:", STRIDE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EMBEDDING_DIM:", EMBEDDING_DIM)
    print("MAX_SEQ_LENGTH:", MAX_SEQ_LENGTH)
    print("NUM_HEADS:", NUM_HEADS)
    print("NUM_LAYERS:", NUM_LAYERS)
    print("DROPOUT:", DROPOUT)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("EPOCHS:", EPOCHS)
    print("Config OK.")
