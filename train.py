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

Every hyperparameter (context length, batch size, learning rate, epochs,
device) comes from config.py.

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
from model import MiniLLM, get_device


# Print a running train loss every this many steps, so long epochs on
# CPU don't look frozen.
LOG_EVERY = 50


def compute_loss(model, X, Y, loss_fn):
    """Forward pass + cross-entropy loss."""
    logits = model(X)  # (batch, seq_len, vocab_size)
    B, T, V = logits.shape
    return loss_fn(logits.view(B * T, V), Y.view(B * T))


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """One pass over the train dataloader. Returns average train loss."""
    model.train()
    total_loss, num_batches = 0.0, 0

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        loss = compute_loss(model, X, Y, loss_fn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if num_batches % LOG_EVERY == 0:
            print(f"      step {num_batches:>5}/{len(dataloader)} | "
                  f"avg train loss {total_loss / num_batches:.4f}", flush=True)

    return total_loss / num_batches


def evaluate(model, dataloader, loss_fn, device):
    """Average loss over the val dataloader, no training. Returns val loss."""
    model.eval()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            loss = compute_loss(model, X, Y, loss_fn)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    torch.manual_seed(0)
    device = get_device()

    text = load_corpus()
    train_text, val_text = train_val_split(text)

    tokenizer = BPETokenizer()
    tokenizer.train(train_text)
    vocab_size = tokenizer.vocab_size

    train_loader = create_dataloader(train_text, tokenizer)
    # Validation should be deterministic and cover every window, so no
    # shuffling and keep the last partial batch.
    val_loader = create_dataloader(val_text, tokenizer, shuffle=False, drop_last=False)

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
    model = MiniLLM(**model_config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Params: {num_params:,} | Vocab size: {vocab_size}")
    print(f"Context: {config.CONTEXT_LENGTH} | Batch size: {config.BATCH_SIZE} | "
          f"LR: {config.LEARNING_RATE} | Train batches/epoch: {len(train_loader)}\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}")
    print("-" * 33)

    # Only keep the epoch with the lowest val loss: later epochs can start
    # overfitting, and the best generalizing model is the one worth keeping.
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, model_config, tokenizer, epoch, val_loss)
            marker = "  <- saved"
        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f}{marker}", flush=True)

    print(f"\nTraining complete. Best val loss {best_val_loss:.4f}, "
          f"checkpoint in {default_checkpoint_dir()}")


if __name__ == "__main__":
    main()
