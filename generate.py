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
    python generate.py --chat --interactive --checkpoint-dir checkpoints/dialog_corpus

Interactive mode loads the model once, then continues each prompt you
type until you enter "quit" (or press Ctrl+C / Ctrl+D). Type \\n for a
line break, e.g.  ROMEO:\\nWhat  -- the model was trained on Shakespeare
laid out as "SPEAKER:" on its own line.

This is text continuation, not chat: the model has only ever seen plays,
so it continues your prompt as if it were a line from one. Each prompt is
continued independently; earlier prompts are not remembered.

Chat mode (--chat) is for a model trained on data/dialog_corpus.txt (see
data/prepare_dialog.py). What you type is wrapped as

    User: <your message>
    Bot:

and the model writes the Bot line. Generation stops at the end of that
line, so only the reply is shown and a short reply comes back quickly.
"""

import argparse

import torch

import config
from checkpoint import load_checkpoint
from model import get_device

DEFAULT_PROMPT = "ROMEO:"
DEFAULT_CHAT_MESSAGE = "Hi, how are you?"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniLLM.")
    parser.add_argument("--prompt", default=None,
                        help="text to continue, or the message to reply to with --chat "
                             f"(default: {DEFAULT_PROMPT!r}, or {DEFAULT_CHAT_MESSAGE!r} "
                             "with --chat)")
    parser.add_argument("--chat", action="store_true",
                        help="reply as the Bot in a User/Bot conversation "
                             "(for a model trained on the dialog corpus)")
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


def generate_ids(model, tokenizer, prompt, args, device, stop_ids=None):
    """Return (prompt_ids, output_ids): the encoded prompt, and prompt + new tokens."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens -- use a non-empty prompt.")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, prompt_len)
    output_ids = model.generate(
        token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
        stop_ids=stop_ids,
    )
    return prompt_ids, output_ids[0].tolist()


def continue_text(model, tokenizer, prompt, args, device):
    """Return `prompt` followed by the model's continuation of it."""
    _, output_ids = generate_ids(model, tokenizer, prompt, args, device)
    return tokenizer.decode(output_ids)


def chat_prompt(message):
    """
    Wrap a message the way the dialog corpus lays out a turn, ending where
    the model should write the reply:

        "hello"  ->  "User: hello\\nBot:"

    Every turn in the corpus is a single line, so line breaks in the
    message become spaces.
    """
    message = " ".join(message.split())
    return f"{config.CHAT_USER_PREFIX} {message}\n{config.CHAT_BOT_PREFIX}"


def newline_token_ids(tokenizer):
    """
    Ids of every token that contains a line break. The BPE tokenizer never
    merges whitespace with letters, so the first token after the end of a
    line is always one of these. Decoding each id (rather than reading a
    vocab dict) works for every tokenizer, GPT2Tokenizer included.
    """
    return {i for i in range(tokenizer.vocab_size) if "\n" in tokenizer.decode([i])}


def chat_reply(model, tokenizer, message, args, device):
    """
    The model's reply to `message`, as the Bot line of a User/Bot
    conversation. Returns (reply, new_tokens). Generation stops at the
    first line break (the end of the Bot's turn) instead of going on to
    write the User's next line, so new_tokens is often well under
    args.max_new_tokens.
    """
    prompt_ids, output_ids = generate_ids(model, tokenizer, chat_prompt(message),
                                          args, device, newline_token_ids(tokenizer))
    new_ids = output_ids[len(prompt_ids):]
    reply = tokenizer.decode(new_ids).split("\n", 1)[0].strip()
    return reply, len(new_ids)


def interactive_loop(model, tokenizer, args, device, read=input, chat=False):
    """
    Read a prompt, print its continuation, repeat. Stops on "quit"/"exit",
    end of input (Ctrl+D, or Ctrl+Z then Enter on Windows) or Ctrl+C.
    `read` is swappable so tests can feed prompts without a keyboard.

    chat=True prints the model's reply to each message instead (see
    chat_reply()).
    """
    if chat:
        print('Type a message and press Enter ("quit" to stop).')
    else:
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

        try:
            if chat:
                reply, _ = chat_reply(model, tokenizer, line, args, device)
                print(f"{config.CHAT_BOT_PREFIX} {reply}")
            else:
                prompt = line.replace("\\n", "\n")
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
        interactive_loop(model, tokenizer, args, device, chat=args.chat)
        return

    try:
        if args.chat:
            message = args.prompt or DEFAULT_CHAT_MESSAGE
            reply, _ = chat_reply(model, tokenizer, message, args, device)
            print(f"{chat_prompt(message)} {reply}")
        else:
            print(continue_text(model, tokenizer, args.prompt or DEFAULT_PROMPT,
                                args, device))
    except ValueError as err:
        raise SystemExit(str(err))


if __name__ == "__main__":
    main()
