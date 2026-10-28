"""
Mini LLM — load OpenAI's pretrained GPT-2 into MiniLLM.

Pipeline:

    Hugging Face hub ("gpt2", downloaded once and cached)
              |
    MiniLLM.from_pretrained_gpt2   (model.py: rename + transpose weights)
              |
    verify: same logits as Hugging Face's own GPT-2 on the same tokens
              |
    measure: val loss / bits per character on Tiny Shakespeare's val split,
             the same numbers train.py prints for our own models
              |
    checkpoints/gpt2/  (checkpoint.py)  -->  generate.py / app.py

Run:

    python load_gpt2.py
    python load_gpt2.py --model gpt2-medium

then use it like any trained checkpoint:

    python generate.py --checkpoint-dir checkpoints/gpt2 --prompt "The meaning of life is"
    python app.py --checkpoint-dir checkpoints/gpt2

GPT-2 is a pretrained base model, not a chatbot: it continues text the
way web pages continue, it doesn't follow instructions. Teaching it to
answer is the fine-tuning step that comes next.

Needs `pip install transformers tiktoken` (transformers only for the
download and the logit check).
"""

import argparse
import math

import torch
import torch.nn as nn

from checkpoint import default_checkpoint_dir, save_checkpoint
from data.corpus import load_corpus, train_val_split
from dataset import create_dataloader
from gpt2_tokenizer import GPT2Tokenizer
from model import MiniLLM, get_device
from train import bits_per_char, evaluate

# Logits from our model and Hugging Face's may differ by float rounding
# (different kernels, different operation order), nothing more. A wrong
# transpose or GELU shows up as differences of 1 or more, not 1e-3.
LOGIT_TOLERANCE = 1e-3

CHECK_TEXT = "Hello, I'm a language model, and I can"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert pretrained GPT-2 into a MiniLLM checkpoint.")
    parser.add_argument("--model", default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="which GPT-2 size (default: %(default)s, 124M parameters)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="where to save (default: checkpoints/<model>/)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="don't measure val loss on Tiny Shakespeare (it takes "
                             "a minute or two on CPU)")
    return parser.parse_args()


@torch.no_grad()
def max_logit_difference(model, name, token_ids):
    """Largest |our logit - Hugging Face's logit| over every position and token."""
    from transformers import GPT2LMHeadModel

    reference = GPT2LMHeadModel.from_pretrained(name).eval()
    expected = reference(token_ids).logits
    return (model(token_ids) - expected).abs().max().item()


def shakespeare_val_loss(model, tokenizer, device):
    """
    GPT-2's loss on the same held-out text train.py validates on, so the
    bits per character can be compared directly with our own models'.
    The tokenizer differs, so loss per *token* can't be compared; bits
    per character can (see train.bits_per_char).
    """
    _, val_text = train_val_split(load_corpus())
    loader = create_dataloader(val_text, tokenizer, context_length=model.max_seq_length,
                               stride=model.max_seq_length, batch_size=4,
                               shuffle=False, drop_last=False)
    val_loss = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    chars_per_token = len(val_text) / len(tokenizer.encode(val_text))
    return val_loss, bits_per_char(val_loss, chars_per_token)


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir() / args.model
    device = get_device()

    print(f"Loading {args.model} weights into MiniLLM...")
    model, model_config = MiniLLM.from_pretrained_gpt2(args.model)
    model.eval()
    tokenizer = GPT2Tokenizer()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {num_params:,} | Vocab: {tokenizer.vocab_size} | "
          f"Context: {model.max_seq_length}")

    check_ids = torch.tensor([tokenizer.encode(CHECK_TEXT)])
    diff = max_logit_difference(model, args.model, check_ids)
    print(f"Max logit difference vs Hugging Face: {diff:.2e}")
    if diff > LOGIT_TOLERANCE:
        raise SystemExit(f"Conversion is wrong: logits differ by {diff:.3f} "
                         f"(tolerance {LOGIT_TOLERANCE}). Not saving.")

    model.to(device)
    if args.skip_eval:
        val_loss = 0.0  # no measurement; generate.py and app.py just display it
    else:
        print("Measuring loss on the Tiny Shakespeare val split...")
        val_loss, val_bpc = shakespeare_val_loss(model, tokenizer, device)
        print(f"Val loss {val_loss:.4f} | perplexity {math.exp(val_loss):.2f} | "
              f"{val_bpc:.4f} bits/char")

    torch.manual_seed(0)
    sample = model.generate(check_ids.to(device), 40, temperature=0.8, top_k=40)
    print(f"\nSample:\n  {tokenizer.decode(sample[0].tolist())!r}\n")

    # epoch 0: this model hasn't been trained by us (yet).
    save_checkpoint(model, model_config, tokenizer, epoch=0, val_loss=val_loss,
                    checkpoint_dir=checkpoint_dir)
    print(f"Saved to {checkpoint_dir}")
    print(f"Try: python generate.py --checkpoint-dir {checkpoint_dir} "
          f"--prompt \"The meaning of life is\"")


if __name__ == "__main__":
    main()
