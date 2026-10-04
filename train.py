"""
Mini LLM — training entry point.

Pipeline:

    Tiny Shakespeare (data/corpus.py)
              |
    BPE tokenizer (bpe_tokenizer.py)
              |
    TextDataset / DataLoader (dataset.py)  -->  Input X / Target Y
              |
            MiniLLM            (model.py)
              |
            Logits
              |
      CrossEntropyLoss  -->  backward()  -->  AdamW.step()
              |
    Best model (lowest val loss) + tokenizer  -->  checkpoints/  (checkpoint.py)

Run:

    python train.py
"""

import torch
import torch.nn as nn

import config
from data.corpus import load_corpus, train_val_split
from bpe_tokenizer import BPETokenizer
from checkpoint import save_checkpoint, default_checkpoint_dir
from dataset import create_dataloader
from model import MiniLLM


BATCH_SIZE = getattr(config, "BATCH_SIZE", 32)
NUM_EPOCHS = 5


def compute_loss(model, X, Y, loss_fn):
    """Forward pass + cross-entropy loss."""
    logits = model(X)  # (batch, seq_len, vocab_size)
    B, T, V = logits.shape
    return loss_fn(logits.view(B * T, V), Y.view(B * T))


def train_epoch(model, dataloader, loss_fn, optimizer):
    """One pass over the train dataloader. Returns average train loss."""
    model.train()
    total_loss, num_batches = 0.0, 0

    for X, Y in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, X, Y, loss_fn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, loss_fn):
    """Average loss over the val dataloader, no training. Returns val loss."""
    model.eval()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for X, Y in dataloader:
            loss = compute_loss(model, X, Y, loss_fn)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    torch.manual_seed(0)

    text = load_corpus()
    train_text, val_text = train_val_split(text)

    tokenizer = BPETokenizer()
    tokenizer.train(train_text)
    vocab_size = tokenizer.vocab_size

    train_loader = create_dataloader(train_text, tokenizer, batch_size=BATCH_SIZE)
    val_loader = create_dataloader(val_text, tokenizer, batch_size=BATCH_SIZE)

    # Stored in the checkpoint so generate.py can rebuild the exact same
    # architecture even if config.py has changed since training.
    model_config = {
        "vocab_size": vocab_size,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "dropout": config.DROPOUT,
    }
    model = MiniLLM(**model_config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Vocab size: {vocab_size} | Batch size: {BATCH_SIZE} | "
          f"Train batches/epoch: {len(train_loader)}\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}")
    print("-" * 33)

    # Only keep the epoch with the lowest val loss: later epochs can start
    # overfitting, and the best generalizing model is the one worth keeping.
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, model_config, tokenizer, epoch, val_loss)
            marker = "  <- saved"
        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f}{marker}")

    print(f"\nTraining complete. Best val loss {best_val_loss:.4f}, "
          f"checkpoint in {default_checkpoint_dir()}")


if __name__ == "__main__":
    main()
