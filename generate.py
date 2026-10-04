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
    python generate.py --interactive

Interactive mode loads the model once, then continues each prompt you
type until you enter "quit" (or press Ctrl+C / Ctrl+D). Type \\n for a
line break, e.g.  ROMEO:\\nWhat  -- the model was trained on Shakespeare
laid out as "SPEAKER:" on its own line.

This is text continuation, not chat: the model has only ever seen plays,
so it continues your prompt as if it were a line from one. Each prompt is
continued independently; earlier prompts are not remembered.
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
    parser.add_argument("--interactive", action="store_true",
                        help="keep prompting for text to continue until 'quit'")
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


def continue_text(model, tokenizer, prompt, args, device):
    """Return `prompt` followed by the model's continuation of it."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens -- use a non-empty prompt.")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, prompt_len)
    output_ids = model.generate(
        token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    )
    return tokenizer.decode(output_ids[0].tolist())


def interactive_loop(model, tokenizer, args, device, read=input):
    """
    Read a prompt, print its continuation, repeat. Stops on "quit"/"exit",
    end of input (Ctrl+D, or Ctrl+Z then Enter on Windows) or Ctrl+C.
    `read` is swappable so tests can feed prompts without a keyboard.
    """
    print('Type a prompt and press Enter ("quit" to stop, \\n for a new line).')
    while True:
        try:
            line = read("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.strip().lower() in {"quit", "exit"}:
            break
        if not line.strip():
            continue

        prompt = line.replace("\\n", "\n")
        try:
            print(continue_text(model, tokenizer, prompt, args, device))
        except ValueError as err:
            print(err)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    model, tokenizer, info = load_checkpoint(args.checkpoint_dir, map_location=device)
    print(f"Loaded checkpoint from epoch {info['epoch']} "
          f"(val loss {info['val_loss']:.4f})\n")

    if args.interactive:
        interactive_loop(model, tokenizer, args, device)
        return

    try:
        print(continue_text(model, tokenizer, args.prompt, args, device))
    except ValueError as err:
        raise SystemExit(str(err))


if __name__ == "__main__":
    main()
