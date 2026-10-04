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
# How many tokens the model sees per training example. Was 8 (a few
# words), which is too short to learn anything beyond the next word or
# two. 128 BPE tokens is ~280 characters of Tiny Shakespeare -- several
# lines of dialogue, enough to pick up speaker/line structure. Training
# cost grows faster than linearly with this, so raise it with care on CPU.
CONTEXT_LENGTH = 128

# How far the sliding window moves each step. stride < context_length
# means overlapping windows (more training samples from the same text);
# stride == context_length means no overlap. Half the context gives every
# token two different positions in the window per epoch.
STRIDE = CONTEXT_LENGTH // 2

# Samples per training batch.
BATCH_SIZE = 32

# Fraction of the corpus held out for validation (by character count, on
# the raw text before tokenizing). 0.1 = 90/10 train/val split -- a
# reasonable default for a first language-modeling pass. A contiguous
# split (not shuffled) is used -- see data/corpus.py's train_val_split()
# -- since shuffling would let the model "see" future context out of
# order and defeats the point of a held-out set for a language model.
VAL_RATIO = 0.1

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
# CONTEXT_LENGTH. Equal to CONTEXT_LENGTH: the model never trains on
# positions past the training window, and generation crops its input to
# the last MAX_SEQ_LENGTH tokens anyway (MiniLLM.generate).
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
# Training (train.py)
# ---------------------------------------------------------------------------
# 1e-3 is a common choice for a model this small; 3e-4 is the usual
# default for larger GPTs but learns slowly given how few steps we run.
LEARNING_RATE = 1e-3
EPOCHS = 10

# Learning-rate schedule: ramp linearly from ~0 to LEARNING_RATE over the
# first WARMUP_STEPS optimizer steps (large updates on freshly initialized
# weights are what destabilize early training), then follow a cosine
# curve down to MIN_LR by the final step, so the last epochs make small,
# careful updates instead of bouncing around the minimum.
WARMUP_STEPS = 100
MIN_LR = LEARNING_RATE / 10

# Rescale the gradients whenever their global L2 norm exceeds this, so a
# single bad batch can't produce one huge destabilizing update.
GRAD_CLIP = 1.0

# Evaluate on the val set (and save a checkpoint if it improved) every
# this many optimizer steps, instead of only once per epoch.
EVAL_EVERY = 100

# A short sample printed at every evaluation, to watch the text improve.
SAMPLE_PROMPT = "ROMEO:"
SAMPLE_TOKENS = 40

# "auto" picks CUDA, then Apple MPS, then CPU. Set to "cpu" / "cuda" /
# "mps" to force one.
DEVICE = "auto"

# ---------------------------------------------------------------------------
# Data paths (data/corpus.py)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
CORPUS_FILENAME = "tiny_corpus.txt"

# ---------------------------------------------------------------------------
# Checkpoints (checkpoint.py)
# ---------------------------------------------------------------------------
# Where train.py saves the best model (lowest val loss) and the tokenizer
# it was trained with, so generate.py can load both without retraining.
# Resolved relative to the project root, like DATA_DIR.
CHECKPOINT_DIR = "checkpoints"
MODEL_CHECKPOINT_FILENAME = "mini_llm.pt"
TOKENIZER_FILENAME = "tokenizer.json"

# ---------------------------------------------------------------------------
# Generation (generate.py)
# ---------------------------------------------------------------------------
# How many new tokens to sample after the prompt.
MAX_NEW_TOKENS = 100

# Divides the logits before softmax. < 1.0 makes sampling more
# conservative, > 1.0 more random, 0 means greedy (always the argmax).
TEMPERATURE = 0.8

# Only sample from the k most likely tokens (None = full vocabulary).
TOP_K = 40


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
    print("VAL_RATIO:", VAL_RATIO)
    print("EMBEDDING_DIM:", EMBEDDING_DIM)
    print("MAX_SEQ_LENGTH:", MAX_SEQ_LENGTH)
    print("NUM_HEADS:", NUM_HEADS)
    print("NUM_LAYERS:", NUM_LAYERS)
    print("DROPOUT:", DROPOUT)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("EPOCHS:", EPOCHS)
    print("WARMUP_STEPS:", WARMUP_STEPS)
    print("MIN_LR:", MIN_LR)
    print("GRAD_CLIP:", GRAD_CLIP)
    print("EVAL_EVERY:", EVAL_EVERY)
    print("DEVICE:", DEVICE)
    print("Config OK.")
