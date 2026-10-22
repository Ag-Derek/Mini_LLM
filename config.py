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
transformer -> training -> checkpoints -> generation.
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
# Number of BPE merge operations to learn during training. Was 500, which
# leaves most words split into pieces (~2.45 characters per token on the
# dialog corpus). Measured on the dialog corpus's held-out text:
#
#     merges   chars/token   text in a 128-token window   model params
#        500          2.45                    313 chars          4.92M
#      2,000          3.09                    395 chars          5.30M
#      4,000          3.33                    426 chars          5.82M
#      8,000          3.49                    447 chars          6.84M
#
# 4,000 gets most of the gain: past it, new tokens are rare whole words the
# model sees too seldom to learn well, each costing EMBEDDING_DIM params.
NUM_MERGES = 4000

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
# Was 128 (~1.3M parameters, ~1.1 s per training step on a 4-thread CPU);
# 256 gives ~4.9M parameters at ~2.8 s per step. Parameters grow with the
# square of this, so doubling it roughly quadruples the model.
EMBEDDING_DIM = 256

# ---------------------------------------------------------------------------
# Attention / Transformer (attention.py, transformer.py, model.py)
# ---------------------------------------------------------------------------
# Longest sequence the model can ever be asked to attend over. This sizes
# the causal mask buffer in CausalSelfAttentionHead, so it must be >=
# CONTEXT_LENGTH. Equal to CONTEXT_LENGTH: the model never trains on
# positions past the training window, and generation crops its input to
# the last MAX_SEQ_LENGTH tokens anyway (MiniLLM.generate).
MAX_SEQ_LENGTH = CONTEXT_LENGTH

# Number of parallel attention heads. EMBEDDING_DIM must be divisible by
# this (asserted in MultiHeadAttention). 8 heads keeps each head at 32
# dimensions, the same as the earlier 128 / 4 setup.
NUM_HEADS = 8

# Number of stacked Transformer blocks (transformer.py's TransformerBlocks).
NUM_LAYERS = 6

# Share the token-embedding matrix with the output layer (lm_head)
# instead of learning two separate (vocab_size, EMBEDDING_DIM) matrices.
# Saves ~146K parameters at the current vocab/embedding size.
TIE_WEIGHTS = True

# Dropout probability, applied to the embeddings (model.py), after
# attention and inside the feed-forward network (transformer.py). Raised
# from 0.1 to 0.2 along with the bigger model: ~4.9M parameters can start
# memorizing a ~1 MB corpus, and stronger dropout pushes back on that
# (nanoGPT uses 0.2 for a similar-sized model on Tiny Shakespeare).
DROPOUT = 0.2

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

# AdamW settings, following GPT-2/3. Weight decay pulls weights toward
# zero a little every step, which discourages memorizing the training
# text. It only applies to weight matrices (embeddings, Linear weights):
# decaying LayerNorm gains and biases just fights the job they do.
# beta2 = 0.95 (instead of the default 0.999) lets Adam's per-weight
# step-size estimate adapt faster, which is more stable for transformers.
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)

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

# Conversation corpus written by data/prepare_dialog.py. Train on it with
#   python train.py --corpus data/dialog_corpus.txt
DIALOG_CORPUS_FILENAME = "dialog_corpus.txt"

# ---------------------------------------------------------------------------
# Chat format (data/prepare_dialog.py, generate.py, app.py)
# ---------------------------------------------------------------------------
# Each turn of a conversation is one line starting with a speaker label,
# and conversations are separated by a blank line:
#
#     User: Hi, how are you?
#     Bot: Fine, thanks. And you?
#
# The labels must match between the corpus and chat mode: chat mode wraps
# what you type as "User: <text>\nBot:" so the model writes the Bot line.
CHAT_USER_PREFIX = "User:"
CHAT_BOT_PREFIX = "Bot:"

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
    print("TIE_WEIGHTS:", TIE_WEIGHTS)
    print("DROPOUT:", DROPOUT)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("EPOCHS:", EPOCHS)
    print("WARMUP_STEPS:", WARMUP_STEPS)
    print("MIN_LR:", MIN_LR)
    print("GRAD_CLIP:", GRAD_CLIP)
    print("EVAL_EVERY:", EVAL_EVERY)
    print("DEVICE:", DEVICE)
    print("Config OK.")
