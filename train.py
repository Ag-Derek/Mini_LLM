"""
Mini LLM — training entry point.

Pipeline:

    Tiny Shakespeare (data/corpus.py)
              |
    BPE tokenizer (bpe_tokenizer.py)   <-- reused from checkpoints/ if present
              |
    TextDataset / DataLoader (dataset.py)  -->  Input X / Target Y
              |
            MiniLLM            (model.py)
              |
            Logits
              |
      CrossEntropyLoss  -->  backward()  -->  clip grads  -->  AdamW.step()
              |                                                  ^
              |                       warmup + cosine LR schedule
              |
    every EVAL_EVERY steps: val loss, perplexity, bits per character, a sample
              |
    Best model (lowest val loss) + tokenizer  -->  checkpoints/  (checkpoint.py)

Every hyperparameter (context length, batch size, learning rate schedule,
epochs, device) comes from config.py.

Run:

    python train.py
    python train.py --corpus data/dialog_corpus.txt --sample-prompt "User: hi\\nBot:"

With --corpus and no --checkpoint-dir, the run saves to
checkpoints/<corpus file name>/ (e.g. checkpoints/dialog_corpus/), so
training on a new corpus never overwrites the Tiny Shakespeare model in
checkpoints/.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

import config
from data.corpus import load_corpus, train_val_split
from bpe_tokenizer import BPETokenizer
from checkpoint import save_checkpoint, default_checkpoint_dir
from dataset import create_dataloader
from model import MiniLLM, get_device


def compute_loss(model, X, Y, loss_fn):
    """Forward pass + cross-entropy loss."""
    logits = model(X)  # (batch, seq_len, vocab_size)
    B, T, V = logits.shape
    return loss_fn(logits.view(B * T, V), Y.view(B * T))


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


def bits_per_char(loss_per_token, chars_per_token):
    """
    Convert a cross-entropy loss in nats per token to bits per character.

    Loss per token depends on the tokenizer: with more BPE merges each
    token covers more text, so per-token loss rises even if the model
    predicts the text equally well. Dividing by the average characters
    per token gives a loss per character, comparable across tokenizers
    (and with character-level models); dividing by ln 2 turns nats into
    bits, the usual unit for this number.
    """
    return loss_per_token / chars_per_token / math.log(2)


def get_lr(step, total_steps):
    """
    Learning rate for optimizer step `step` (0-based): linear warmup to
    config.LEARNING_RATE over config.WARMUP_STEPS, then cosine decay to
    config.MIN_LR at total_steps.

        lr
        |    ____
        |   /    ```--..__
        |  /              ``--..__
        | /                       ``-- MIN_LR
        +---------------------------------> step
          warmup        cosine decay
    """
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / config.WARMUP_STEPS

    progress = (step - config.WARMUP_STEPS) / max(1, total_steps - config.WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))  # 1 -> 0
    return config.MIN_LR + cosine * (config.LEARNING_RATE - config.MIN_LR)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a MiniLLM.")
    parser.add_argument("--corpus", default=None,
                        help="path to a UTF-8 text file to train on "
                             "(default: config.DATA_DIR / config.CORPUS_FILENAME)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="where to save the model and tokenizer (default: "
                             "config.CHECKPOINT_DIR, or checkpoints/<corpus name>/ "
                             "when --corpus is given)")
    parser.add_argument("--sample-prompt", default=config.SAMPLE_PROMPT,
                        help="prompt for the samples printed during training "
                             "(default: %(default)r); \\n means a line break")
    args = parser.parse_args(argv)
    args.sample_prompt = args.sample_prompt.replace("\\n", "\n")
    return args


def resolve_checkpoint_dir(corpus, checkpoint_dir):
    """
    Where this run saves. An explicit --checkpoint-dir wins; otherwise a
    custom corpus gets its own folder next to the default checkpoint, named
    after the corpus file, so different corpora never share (and overwrite)
    a model or tokenizer.
    """
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    if corpus is not None:
        return default_checkpoint_dir() / Path(corpus).stem
    return default_checkpoint_dir()


def load_or_train_tokenizer(train_text, checkpoint_dir=None):
    """
    Reuse the tokenizer saved in checkpoint_dir (defaults to
    config.CHECKPOINT_DIR) by a previous run (BPE
    training takes ~50s on the full corpus), otherwise train a new one.

    The saved tokenizer is only reused if it has the configured number
    of merges and reproduces the training text exactly; anything else
    (NUM_MERGES changed, a tokenizer saved by an older version of
    bpe_tokenizer.py) triggers retraining. Delete checkpoints/ to force
    a fresh tokenizer.
    """
    path = Path(checkpoint_dir or default_checkpoint_dir()) / config.TOKENIZER_FILENAME
    if path.exists():
        tokenizer = BPETokenizer.load(path)
        probe = train_text[:10_000]
        if (len(tokenizer.merges) == config.NUM_MERGES
                and tokenizer.decode(tokenizer.encode(probe)) == probe):
            print(f"Reusing tokenizer from {path}")
            return tokenizer
        print(f"Saved tokenizer at {path} doesn't match config, retraining.")

    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, num_merges=config.NUM_MERGES)
    return tokenizer


def sample_text(model, tokenizer, device, prompt=config.SAMPLE_PROMPT):
    """Short continuation of `prompt` from the current model."""
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    model.eval()
    output_ids = model.generate(prompt_ids, config.SAMPLE_TOKENS,
                                temperature=config.TEMPERATURE, top_k=config.TOP_K)
    return tokenizer.decode(output_ids[0].tolist())


def main():
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.corpus, args.checkpoint_dir)

    torch.manual_seed(0)
    device = get_device()

    text = load_corpus(args.corpus)
    train_text, val_text = train_val_split(text)

    tokenizer = load_or_train_tokenizer(train_text, checkpoint_dir)
    vocab_size = tokenizer.vocab_size

    train_loader = create_dataloader(train_text, tokenizer)
    # Validation should be deterministic and cover every window, so no
    # shuffling and keep the last partial batch.
    val_loader = create_dataloader(val_text, tokenizer, shuffle=False, drop_last=False)
    val_chars_per_token = len(val_text) / len(tokenizer.encode(val_text))

    # Stored in the checkpoint so generate.py can rebuild the exact same
    # architecture even if config.py has changed since training.
    model_config = {
        "vocab_size": vocab_size,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "dropout": config.DROPOUT,
        "tie_weights": config.TIE_WEIGHTS,
    }
    model = MiniLLM(**model_config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.optimizer_param_groups(config.WEIGHT_DECAY),
                                  lr=config.LEARNING_RATE, betas=config.BETAS)

    total_steps = config.EPOCHS * len(train_loader)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Corpus: {args.corpus or config.CORPUS_FILENAME} ({len(text):,} chars) | "
          f"Saving to: {checkpoint_dir}")
    print(f"Device: {device} | Params: {num_params:,} | Vocab size: {vocab_size}")
    print(f"Context: {config.CONTEXT_LENGTH} | Batch size: {config.BATCH_SIZE} | "
          f"Peak LR: {config.LEARNING_RATE} | Epochs: {config.EPOCHS} | "
          f"Steps: {total_steps} ({len(train_loader)}/epoch)\n")
    print(f"{'Step':>6} | {'Epoch':>5} | {'LR':>8} | {'Train Loss':>10} | "
          f"{'Val Loss':>8} | {'Val PPL':>8} | {'Val BPC':>7}")
    print("-" * 72)

    # Only keep the lowest val loss: later steps can start overfitting,
    # and the best generalizing model is the one worth keeping.
    best_val_loss = float("inf")
    best_val_bpc = float("inf")
    running_loss, running_steps = 0.0, 0
    step = 0

    model.train()
    for epoch in range(1, config.EPOCHS + 1):
        for X, Y in train_loader:
            lr = get_lr(step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, X, Y, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            step += 1
            running_loss += loss.item()
            running_steps += 1

            if step % config.EVAL_EVERY != 0 and step != total_steps:
                continue

            # --- periodic evaluation ---------------------------------
            # Train loss is averaged over the steps since the last eval,
            # so it tracks the current model rather than the whole run.
            train_loss = running_loss / running_steps
            running_loss, running_steps = 0.0, 0

            val_loss = evaluate(model, val_loader, loss_fn, device)
            # Perplexity = exp(loss): roughly how many tokens the model
            # is choosing between at each step (vocab_size = pure guessing).
            val_ppl = math.exp(val_loss)
            val_bpc = bits_per_char(val_loss, val_chars_per_token)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_bpc = val_bpc
                save_checkpoint(model, model_config, tokenizer, epoch, val_loss,
                                checkpoint_dir)
                marker = "  <- saved"

            print(f"{step:>6} | {epoch:>5} | {lr:>8.2e} | {train_loss:>10.4f} | "
                  f"{val_loss:>8.4f} | {val_ppl:>8.2f} | {val_bpc:>7.4f}{marker}")
            sample = sample_text(model, tokenizer, device, args.sample_prompt)
            print("         " + sample.replace("\n", "\n         ") + "\n", flush=True)

            model.train()

    print(f"Training complete. Best val loss {best_val_loss:.4f} "
          f"(perplexity {math.exp(best_val_loss):.2f}, "
          f"{best_val_bpc:.4f} bits/char), "
          f"checkpoint in {checkpoint_dir}")


if __name__ == "__main__":
    main()
