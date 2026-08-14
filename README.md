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
Next Token Prediction (568-way logits)
    │
    ▼
Training (CrossEntropyLoss + AdamW)
    │
    ▼
Generated Text  (next up)
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
| Text Generation                     | ⏳ In Progress |

Current progress places the project at approximately **85% completion** toward a fully functional GPT-style language model — the architecture and training pipeline are both wired end to end; generation is the remaining piece.

---

# Features

## Tokenization

* Character-level tokenizer
* Word-level tokenizer
* Byte Pair Encoding (BPE), trained on the real corpus (568-token vocabulary at 500 merges)
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
Linear LM Head (embedding_dim → vocab_size)
    ↓
Logits
```

Current configuration:

| Setting              | Value       |
| --------------------- | ----------- |
| Vocabulary size       | 568         |
| Embedding dimension   | 128         |
| Attention heads        | 4           |
| Dimensions per head    | 32          |
| Transformer layers    | 6           |
| Max sequence length   | 8           |
| Dropout                | 0.1         |
| FFN expansion          | 4×          |
| Total parameters       | ~1.33M      |

The end-to-end forward pass has been verified — a batch of token IDs of shape `(batch, seq_len)` produces logits of shape `(batch, seq_len, vocab_size)`.

---

## Training (`train.py`)

A training loop that:

* Loads and tokenizes the Tiny Shakespeare corpus
* Builds train and validation `DataLoader`s
* Instantiates `MiniLLM`, `CrossEntropyLoss`, and the `AdamW` optimizer
* Runs a configurable number of epochs, drawing a fresh batch every step
* Reports training loss and validation loss once per epoch, so overfitting is visible as it happens

The pipeline was first verified with a single-batch overfit test (training repeatedly on one fixed batch), which confirmed loss collapses from the random-guess baseline (`ln(vocab_size) ≈ 6.34`) toward zero — proving gradients flow correctly through the entire computation graph before committing to a full training run.

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
├── generate.py
├── config.py
├── utils.py
└── README.md
```

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

* Python
* PyTorch
* NumPy
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
* ✅ Training Pipeline (`train.py`, train/val loss per epoch)

## In Progress

* ⏳ Text Generation (autoregressive sampling)

## Planned

* ⬜ Model Evaluation
* ⬜ Model Checkpointing
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