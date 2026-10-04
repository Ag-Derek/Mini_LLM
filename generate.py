"""
Mini LLM — text generation entry point.

Pipeline:

    checkpoints/  (saved by train.py)
          |
    MiniLLM + BPE tokenizer   (checkpoint.py)
          |
    Prompt text  -->  tokenizer.encode  -->  token ids
          |
    model.generate()  -- predict next token, append, repeat --
          |
    tokenizer.decode  -->  generated text

Run (after `python train.py`):

    python generate.py
    python generate.py --prompt "JULIET:" --max-new-tokens 200 --temperature 1.0
"""

import argparse

import torch

import config
from checkpoint import load_checkpoint
from model import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniLLM.")
    parser.add_argument("--prompt", default="ROMEO:",
                        help="text to continue (default: %(default)r)")
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE,
                        help="0 = greedy, <1 = conservative, >1 = more random")
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help="sample only from the k most likely tokens (0 = all)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="defaults to config.CHECKPOINT_DIR")
    parser.add_argument("--seed", type=int, default=None,
                        help="set for reproducible samples")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    model, tokenizer, info = load_checkpoint(args.checkpoint_dir, map_location=device)
    print(f"Loaded checkpoint from epoch {info['epoch']} "
          f"(val loss {info['val_loss']:.4f})\n")

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("Prompt produced no tokens -- use a non-empty prompt.")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, prompt_len)
    output_ids = model.generate(
        token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    )

    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
