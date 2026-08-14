"""
Mini LLM — training entry point.

Where this fits in the pipeline:

    Tiny Shakespeare (data/corpus.py)
              |
    BPE tokenizer (bpe_tokenizer.py)
              |
    TextDataset / DataLoader (dataset.py)
              |
         Input X / Target Y  (Y = X shifted by one token)
              |
            MiniLLM            <-- model.py
              |
            Logits
              |
      CrossEntropyLoss
              |
       loss.backward()
              |
       AdamW.step()

This file is deliberately structured with small, separable pieces so the
single-batch overfit test can later be swapped for a full training loop
without a rewrite:

    get_batch()                  - pull one batch from a dataloader
    compute_loss()                - forward pass + CrossEntropyLoss
    train_step()                  - one forward + backward + optimizer step,
                                     returns (loss, grad_norm)
    single_batch_overfit_test()   - repeatedly train on ONE fixed batch
    main()

Run this file directly to execute the single-batch overfit test:

    python train.py

Expected result: loss should fall from roughly ln(vocab_size) (~6.34 for
vocab_size=568) toward near-zero over a few hundred steps, since the model
is being asked to memorize a single fixed batch. See the module docstring
in the chat discussion for how to read the loss/grad-norm curve if it
DOESN'T behave that way (flat, slow, or exploding).
"""

import torch
import torch.nn as nn

import config
from data.corpus import load_corpus, train_val_split
from bpe_tokenizer import BPETokenizer
from dataset import create_dataloader
from model import MiniLLM


# ---------------------------------------------------------------------------
# 1. Batch fetching
# ---------------------------------------------------------------------------

def get_batch(dataloader):
    """
    Pull a single (X, Y) batch from a dataloader.

    X: (batch, seq_len)   input token ids
    Y: (batch, seq_len)   target token ids, i.e. X shifted by one position
    """
    return next(iter(dataloader))


# ---------------------------------------------------------------------------
# 2. Loss computation
# ---------------------------------------------------------------------------

def compute_loss(model, X, Y, loss_fn):
    """
    Forward pass + cross-entropy loss.

    X: (batch, seq_len)
    Y: (batch, seq_len)
    Returns: scalar loss tensor
    """
    logits = model(X)  # (batch, seq_len, vocab_size)

    # CrossEntropyLoss expects (N, C) predictions vs (N,) targets, so we
    # flatten the batch and sequence dimensions together.
    B, T, V = logits.shape
    loss = loss_fn(logits.view(B * T, V), Y.view(B * T))
    return loss


# ---------------------------------------------------------------------------
# 3. One training step
# ---------------------------------------------------------------------------

def train_step(model, X, Y, loss_fn, optimizer):
    """
    One forward + backward + optimizer step.

    Returns: (loss_value: float, grad_norm: float)

    grad_norm is the total L2 norm across all parameter gradients, computed
    AFTER backward() but BEFORE step() (and before the gradients are
    cleared) so it reflects exactly what the optimizer is about to apply.
    """
    optimizer.zero_grad()

    loss = compute_loss(model, X, Y, loss_fn)
    loss.backward()

    grad_norm = torch.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in model.parameters()
            if p.grad is not None)
    ).item()

    optimizer.step()

    return loss.item(), grad_norm


# ---------------------------------------------------------------------------
# 4. Single-batch overfit test
# ---------------------------------------------------------------------------

def single_batch_overfit_test(model, X, Y, loss_fn, optimizer,
                               num_steps=500, print_every=25):
    """
    Repeatedly train on the SAME fixed (X, Y) batch.

    This isolates the training pipeline (forward -> loss -> backward ->
    optimizer step -> lower loss) from questions about data or
    generalization. The model has no excuse not to memorize one batch;
    if loss doesn't collapse toward ~0, something upstream is broken.
    """
    print(f"Overfitting a single batch of shape X={tuple(X.shape)}, "
          f"Y={tuple(Y.shape)} for {num_steps} steps.\n")

    for step in range(num_steps + 1):
        loss_value, grad_norm = train_step(model, X, Y, loss_fn, optimizer)

        if step % print_every == 0 or step == num_steps:
            print(f"Step {step:4d} | Loss: {loss_value:.4f} | "
                  f"Grad norm: {grad_norm:.4f}")

    return loss_value


# ---------------------------------------------------------------------------
# Demo / entry point
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)

    # --- Data: load corpus, train tokenizer on the train split ---
    text = load_corpus()
    train_text, val_text = train_val_split(text)

    tokenizer = BPETokenizer()
    tokenizer.train(train_text)
    vocab_size = len(tokenizer.vocab)

    train_loader = create_dataloader(train_text, tokenizer)

    # --- Model, loss, optimizer ---
    model = MiniLLM(vocab_size=vocab_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Vocab size: {vocab_size} | Random-guess baseline loss "
          f"(ln(vocab_size)): {torch.log(torch.tensor(float(vocab_size))):.4f}\n")

    # --- Fixed batch, pulled once and reused every step ---
    X, Y = get_batch(train_loader)

    final_loss = single_batch_overfit_test(
        model, X, Y, loss_fn, optimizer,
        num_steps=500, print_every=25,
    )

    print(f"\nFinal loss after overfitting: {final_loss:.4f}")
    if final_loss < 0.1:
        print("PASS — pipeline verified (loss collapsed toward ~0).")
    else:
        print("Loss did not collapse — inspect requires_grad, optimizer "
              "param group, learning rate, and gradient flow.")


if __name__ == "__main__":
    main()
