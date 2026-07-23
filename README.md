# Mini LLM — From-Scratch Language Model Implementation

A from-scratch Mini LLM implementation exploring tokenization, dataset construction, embeddings, attention mechanisms, and Transformer architecture.

## About This Project

Created by **Derrick Agorhom**, this project focuses on understanding the internal mechanics of Large Language Models by implementing core components step-by-step rather than relying entirely on existing frameworks.

The project explores the complete language modeling pipeline:


Raw Text
↓
Tokenization
↓
Token IDs
↓
Training Dataset
↓
Embeddings
↓
Attention Mechanisms
↓
Transformer Architecture
↓
Text Generation


Current implementations include:

- Character-level tokenizer
- Word-level tokenizer
- Byte Pair Encoding (BPE) tokenizer
- Sliding-window dataset generation for next-token prediction

Upcoming components include:

- Token embeddings
- Positional embeddings
- Self-attention mechanisms
- Multi-head attention
- Transformer blocks
- Training loop
- Text generation

The goal is not only to build a working language model, but to understand why modern LLM architectures work by reconstructing their fundamental building blocks from the ground up.
