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

The ultimate goal is to build a fully functional GPT-style language model from scratch.

---

# Project Architecture

The complete data flow is shown below.

```text
Raw Text
    │
    ▼
Character Tokenizer
    │
    ▼
Word Tokenizer
    │
    ▼
Byte Pair Encoding (BPE)
    │
    ▼
Token IDs
    │
    ▼
Dataset Pipeline
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
Multi-Head Self-Attention
    │
    ▼
Feed Forward Network
    │
    ▼
Transformer Blocks
    │
    ▼
Language Model Head
    │
    ▼
Next Token Prediction
    │
    ▼
Generated Text
```

---

# Current Progress

The following components have been implemented from scratch.

| Component                    | Status     |
| ---------------------------- | ---------- |
| Character Tokenizer          | ✅ Complete |
| Word Tokenizer               | ✅ Complete |
| Byte Pair Encoding (BPE)     | ✅ Complete |
| Dataset Pipeline             | ✅ Complete |
| Custom Embedding Layer       | ✅ Complete |
| Positional Embeddings        | ✅ Complete |
| Scaled Dot-Product Attention | ✅ Complete |
| Causal Self-Attention        | ✅ Complete |
| Multi-Head Attention         | ✅ Complete |

Current progress places the project at approximately **60% completion** toward a fully functional GPT-style language model.

---

# Features

## Tokenization

* Character-level tokenizer
* Word-level tokenizer
* Byte Pair Encoding (BPE)
* Automatic vocabulary construction
* Character-to-ID mapping
* ID-to-character mapping
* Unknown token handling
* Lowercase normalization
* Vocabulary serialization
* Save and load functionality

---

## Dataset Pipeline

* Sliding window sequence generation
* Input-target pair creation
* Dataset abstraction using `torch.utils.data.Dataset`
* Mini-batch loading with `DataLoader`

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

# Repository Structure

```text
Mini_LLM/
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
Multi-Head Attention
      │
      ▼
Transformer Block
      │
      ▼
Prediction
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
* Transformer architecture
* Language model training
* Autoregressive text generation

---

# Roadmap

## Completed

* ✅ Character Tokenizer
* ✅ Word Tokenizer
* ✅ Byte Pair Encoding (BPE)
* ✅ Dataset Pipeline
* ✅ Custom Embedding Layer
* ✅ Positional Embeddings
* ✅ Scaled Dot-Product Attention
* ✅ Causal Self-Attention
* ✅ Multi-Head Attention

## In Progress

* ⏳ Feed Forward Network
* ⏳ Layer Normalization
* ⏳ Residual Connections
* ⏳ Transformer Block

## Planned

* ⬜ Complete GPT Model
* ⬜ Language Model Head
* ⬜ Training Pipeline
* ⬜ Model Evaluation
* ⬜ Text Generation
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
