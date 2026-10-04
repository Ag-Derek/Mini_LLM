# Mini LLM – Building a GPT-Style Language Model from Scratch

A complete educational implementation of a **GPT-style Language Model** built from first principles using **PyTorch**.

The objective of this project is not simply to train a language model, but to understand **how modern Large Language Models (LLMs) work internally** by implementing every major component manually instead of relying on high-level libraries.

Every module is built, tested, and verified independently before being integrated into the final model.

---

# Overview

Modern language models appear almost magical—they can answer questions, write code, summarize documents, and generate human-like text.

Under the hood, however, they are built from a collection of surprisingly elegant components.

This project explores those components one by one.

Instead of importing an existing Transformer implementation, we progressively build one ourselves, gaining a deep understanding of:

* How text becomes tokens
* How tokens become embeddings
* Why positional information is required
* How self-attention allows tokens to communicate
* Why multiple attention heads improve learning
* How Transformer blocks are constructed
* How GPT models learn to predict the next token
* How a training loop turns gradients into a model that generalizes

The ultimate goal is to build a fully functional GPT-style language model from scratch.

---

# Project Architecture

The complete data flow is shown below.

```text
Raw Text (Tiny Shakespeare)
    │
    ▼
Byte Pair Encoding (BPE)
    │
    ▼
Token IDs
    │
    ▼
Dataset Pipeline (train / val split)
    │
    ▼
Token Embeddings
    │
    ▼
Positional Embeddings
    │
    ▼
Combined Embeddings
    │
    ▼
Transformer Block × 6
    │  ├── Pre-LayerNorm
    │  ├── Multi-Head Causal Self-Attention
    │  ├── Residual Connection
    │  ├── Pre-LayerNorm
    │  ├── Feed-Forward Network
    │  └── Residual Connection
    ▼
Final LayerNorm
    │
    ▼
Language Model Head
    │
    ▼
Next Token Prediction (569-way logits)
    │
    ▼
Training (CrossEntropyLoss + AdamW)
    │
    ▼
Checkpoint (best val loss)
    │
    ▼
Generated Text  (generate.py)
```

---

# Current Progress

The following components have been implemented from scratch.

| Component                          | Status      |
| ----------------------------------- | ----------- |
| Character Tokenizer                 | ✅ Complete |
| Word Tokenizer                      | ✅ Complete |
| Byte Pair Encoding (BPE)            | ✅ Complete |
| Dataset Pipeline (train/val split)  | ✅ Complete |
| Custom Embedding Layer              | ✅ Complete |
| Positional Embeddings               | ✅ Complete |
| Scaled Dot-Product Attention        | ✅ Complete |
| Causal Self-Attention               | ✅ Complete |
| Multi-Head Attention                | ✅ Complete |
| Feed-Forward Network                | ✅ Complete |
| Pre-Norm Transformer Block          | ✅ Complete |
| 6-Layer Transformer Stack           | ✅ Complete |
| Final LayerNorm + LM Head           | ✅ Complete |
| End-to-End Forward Pass (`model.py`)| ✅ Complete |
| Training Loop (`train.py`)          | ✅ Complete |
| Model Checkpointing                 | ✅ Complete |
| Text Generation (`generate.py`)     | ✅ Complete |

The project is now a **working end-to-end GPT-style language model**: train it, save the best checkpoint, and sample text from it. It trains on 128-token windows with a whitespace-preserving tokenizer, so generated text keeps Shakespeare's line and speaker structure. Training uses a warmup + cosine learning-rate schedule and gradient clipping, and reports validation perplexity and live samples as it goes.

---

# Features

## Tokenization

* Character-level tokenizer
* Word-level tokenizer
* Byte Pair Encoding (BPE), trained on the real corpus (569-token vocabulary at 500 merges)
* GPT-2-style pre-tokenization: words keep their leading space (`" the"`), and newlines are tokens, so `decode(encode(text)) == text` exactly
* Case preserved (not lowercased) so character names and sentence starts carry signal
* Automatic vocabulary construction
* Token-to-ID and ID-to-token mapping
* Unknown token handling
* Vocabulary serialization
* Save and load functionality

---

## Dataset Pipeline

* Real training corpus: Tiny Shakespeare (~1.1M characters)
* Sliding window sequence generation
* Input-target pair creation (target = input shifted by one token)
* 90/10 contiguous train/validation split
* Dataset abstraction using `torch.utils.data.Dataset`
* Mini-batch loading with `DataLoader`
* Centralized configuration via `config.py`

---

## Embeddings

Implemented entirely from scratch.

Features include:

* Learnable embedding matrix
* Random weight initialization
* Efficient embedding lookup
* Sparse gradient updates
* Verification against `torch.nn.Embedding`

---

## Positional Embeddings

Implemented learnable positional embeddings that provide ordering information to the Transformer.

The final embedding is computed as

```text
Token Embedding
        +
Position Embedding
```

allowing the model to distinguish between sentences containing the same words in different orders.

---

## Attention

Implemented from scratch:

* Query projection
* Key projection
* Value projection
* Scaled dot-product attention
* Softmax normalization
* Weighted value aggregation

The implementation closely follows the attention mechanism introduced in the original Transformer architecture.

---

## Causal Self-Attention

Implemented causal masking for autoregressive language modeling.

Future tokens are masked before the softmax operation so the model cannot "peek" at words it is trying to predict.

```text
Dog
↓

Can only attend to Dog

Bites
↓

Can attend to Dog and Bites

Man
↓

Can attend to all previous words
```

---

## Multi-Head Attention

Implemented parallel attention heads that learn different relationships between tokens.

Each head computes its own

* Queries
* Keys
* Values
* Attention matrix

before their outputs are concatenated and projected back into the original embedding dimension.

---

## Feed-Forward Network

Position-wise feed-forward network applied identically at every sequence position:

```text
Linear(embedding_dim → 4 × embedding_dim)
        ↓
      GELU
        ↓
Linear(4 × embedding_dim → embedding_dim)
        ↓
     Dropout
```

Where attention lets tokens exchange information, the feed-forward network transforms each token's representation independently and non-linearly.

---

## Transformer Block

A Pre-Norm Transformer block combining multi-head attention with the feed-forward network:

```text
X1 = X  + Dropout(MHA(LayerNorm(X)))
X2 = X1 + FFN(LayerNorm(X1))
```

Pre-Norm keeps the residual pathway clean, which makes deep stacks of blocks easier to train than the original Post-Norm formulation.

Six of these blocks are stacked to form the full Transformer.

---

## Language Model (`model.py`)

`MiniLLM` wires the full pipeline into a single `nn.Module`:

```text
Token IDs
    ↓
Token Embedding + Position Embedding
    ↓
Transformer Block × 6
    ↓
Final LayerNorm
    ↓
Linear LM Head (embedding_dim → vocab_size, shares the token-embedding matrix)
    ↓
Logits
```

Current configuration:

| Setting              | Value       |
| --------------------- | ----------- |
| Vocabulary size       | 569         |
| Embedding dimension   | 128         |
| Attention heads        | 4           |
| Dimensions per head    | 32          |
| Transformer layers    | 6           |
| Max sequence length   | 128         |
| Dropout                | 0.1         |
| FFN expansion          | 4×          |
| Weight tying           | On          |
| Total parameters       | ~1.28M      |

The end-to-end forward pass has been verified — a batch of token IDs of shape `(batch, seq_len)` produces logits of shape `(batch, seq_len, vocab_size)`.

**Weight tying** (`config.TIE_WEIGHTS`): the LM head reuses the token-embedding matrix instead of learning its own. Both are `(vocab_size, embedding_dim)`, so one matrix serves both directions: a token is predicted when the final hidden state looks like that token's embedding. This saves 72,832 parameters and, over the same 10-epoch run, lowered the best validation loss from 3.308 to 3.256 (perplexity 27.3 → 26.0). Checkpoints saved before this change still load, as untied models.

---

## Training (`train.py`)

A training loop that:

* Loads and tokenizes the Tiny Shakespeare corpus, reusing the tokenizer saved in `checkpoints/` when it matches the config (skips ~50s of BPE training)
* Builds train and validation `DataLoader`s
* Instantiates `MiniLLM`, `CrossEntropyLoss`, and the `AdamW` optimizer
* Runs a configurable number of epochs, drawing a fresh batch every step
* Uses a learning-rate schedule: linear warmup over the first 100 steps to a peak of 1e-3, then cosine decay to 1e-4
* Clips the global gradient norm to 1.0 so one bad batch can't derail training
* Takes every hyperparameter from `config.py` (128-token context, batch size 32, 10 epochs)
* Runs on CUDA, Apple MPS, or CPU automatically (`config.DEVICE = "auto"`)
* Every 100 steps reports train loss, validation loss and **perplexity** (`exp(val_loss)`), plus a short sample from the prompt `ROMEO:`, so you can watch the text improve
* Saves the model and tokenizer to `checkpoints/` whenever validation loss improves (`checkpoint.py`)

```text
  Step | Epoch |       LR | Train Loss | Val Loss |  Val PPL
--------------------------------------------------------------
   300 |     1 | 4.33e-04 |     4.3947 |   4.1325 |    62.33  <- saved
         ROMEO:
         Pay fovedech, sir, Dileed.
```

Perplexity reads as "how many tokens the model is effectively choosing between": 569 means pure guessing over the vocabulary, and lower is better.

The pipeline was first verified with a single-batch overfit test (training repeatedly on one fixed batch), which confirmed loss collapses from the random-guess baseline (`ln(vocab_size) ≈ 6.34`) toward zero — proving gradients flow correctly through the entire computation graph before committing to a full training run.

---

## Text Generation (`generate.py`)

Loads the best checkpoint saved by `train.py` and continues a prompt one token at a time:

```text
Prompt -> encode -> [ crop to last MAX_SEQ_LENGTH tokens -> model -> last-position logits
                      -> temperature / top-k -> sample -> append ] x N -> decode
```

```bash
pip install -r requirements.txt
python train.py      # trains and writes checkpoints/
python generate.py --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

* `--temperature 0` gives greedy decoding; higher values give more varied text
* `--top-k` restricts sampling to the k most likely tokens (`0` = whole vocabulary)
* `--seed` makes a sample reproducible

---

# Repository Structure

```text
Mini_LLM/
│
├── data/
│   ├── tiny_corpus.txt
│   ├── corpus.py
│   └── README.md
│
├── tokenizer.py
├── word_tokenizer.py
├── bpe_tokenizer.py
├── dataset.py
├── embeddings.py
├── attention.py
├── transformer.py
├── model.py
├── train.py
├── checkpoint.py
├── generate.py
├── config.py
│
├── tests/
│   ├── test_tokenizers.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
│
├── checkpoints/        (created by train.py, git-ignored)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Testing

```bash
python -m pytest
```

34 fast tests (~15 seconds on CPU, no trained model needed) cover every stage of the pipeline:

* **Tokenizers** — exact BPE round trip including newlines and indentation, `<unk>` handling, save/load
* **Data** — corpus split, input/target shift, sliding-window stride
* **Model** — `MyEmbedding` vs `nn.Embedding`, causal mask, a check that editing a future token never changes earlier predictions, generation past the context window, greedy decoding
* **Single-batch overfit** — a tiny model memorizes one batch, proving gradients reach every layer
* **Training** — warmup + cosine LR schedule, checkpoint save/reload, tokenizer reuse

---

# Example Learning Pipeline

A simple sentence

```text
Dog bites man
```

passes through the following stages:

```text
Raw Text
      │
      ▼
BPE Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Embedding Layer
      │
      ▼
Positional Embeddings
      │
      ▼
Transformer Block × 6
      │
      ▼
Final LayerNorm + LM Head
      │
      ▼
Next-Token Prediction
```

Each stage transforms the data into increasingly meaningful numerical representations.

---

# Technologies

This project is implemented using:

* Python 3.10+
* PyTorch
* pytest (tests only)
* Git

No high-level Transformer libraries are used.

The objective is to understand every major algorithm by implementing it manually.

---

# Learning Objectives

This project explores the internal mechanics of modern Transformer-based language models.

Topics covered include:

* Tokenization
* Vocabulary construction
* Embedding layers
* Positional embeddings
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* Feed-forward networks
* Residual connections and Pre-Norm Transformer blocks
* Language model training with train/validation splits
* Autoregressive text generation

---

# Roadmap

## Completed

* ✅ Character Tokenizer
* ✅ Word Tokenizer
* ✅ Byte Pair Encoding (BPE)
* ✅ Dataset Pipeline (train/val split)
* ✅ Custom Embedding Layer
* ✅ Positional Embeddings
* ✅ Scaled Dot-Product Attention
* ✅ Causal Self-Attention
* ✅ Multi-Head Attention
* ✅ Feed-Forward Network
* ✅ Pre-Norm Transformer Block
* ✅ 6-Layer Transformer Stack
* ✅ Language Model Head (`model.py`)
* ✅ End-to-End Forward Pass
* ✅ Training Pipeline (`train.py`, train/val loss)
* ✅ Model Checkpointing
* ✅ Text Generation (autoregressive sampling)
* ✅ Learning-Rate Warmup + Cosine Decay, Gradient Clipping
* ✅ Model Evaluation (validation perplexity + live samples)
* ✅ Test Suite (pytest)
* ✅ Weight Tying (shared token embedding / LM head)

## Planned

* ⬜ Interactive Chat Interface

---

# Why Build Everything from Scratch?

Libraries such as Hugging Face and PyTorch provide highly optimized implementations of Transformer models that can be used with only a few lines of code.

While these libraries are invaluable in practice, they often hide the mathematical and algorithmic ideas that make modern language models work.

This project removes that abstraction.

Every component is implemented manually, verified independently, and integrated step by step, making the complete architecture easier to understand, debug, and extend.

The goal is not only to build a language model, but also to develop a deep understanding of the principles behind modern Transformer architectures.

---

# Author

**Derrick Agorhom**

Building a GPT-style language model from scratch to gain a practical understanding of the architecture that powers modern Large Language Models.