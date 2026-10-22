"""
generate.py tests: one-shot continuation, chat replies and the interactive
prompt loop, using a tiny untrained model so no checkpoint is needed.
"""

from types import SimpleNamespace

import pytest
import torch

import config
from bpe_tokenizer import BPETokenizer
from generate import (chat_prompt, chat_reply, continue_text, interactive_loop,
                      newline_token_ids)
from model import MiniLLM

TEXT = "ROMEO:\nTo be, or not to be, that is the question:\n" * 20
CHAT_TEXT = "User: Hi, how are you?\nBot: Fine, thanks.\n\n" * 20
ARGS = SimpleNamespace(max_new_tokens=5, temperature=0, top_k=0)
CHAT_ARGS = SimpleNamespace(max_new_tokens=40, temperature=0, top_k=0)
DEVICE = torch.device("cpu")


@pytest.fixture
def model_and_tokenizer():
    tok = BPETokenizer()
    tok.train(TEXT, num_merges=20)
    torch.manual_seed(0)
    model = MiniLLM(vocab_size=tok.vocab_size, embedding_dim=32, num_heads=4,
                    num_layers=2, max_seq_length=16, dropout=0.0).eval()
    return model, tok


def scripted(lines):
    """A stand-in for input() that returns `lines`, then signals end of input."""
    it = iter(lines)

    def read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return read


def test_continue_text_starts_with_prompt_and_adds_tokens(model_and_tokenizer):
    model, tok = model_and_tokenizer
    out = continue_text(model, tok, "ROMEO:", ARGS, DEVICE)
    assert out.startswith("ROMEO:")
    assert len(tok.encode(out)) == len(tok.encode("ROMEO:")) + ARGS.max_new_tokens


def test_continue_text_rejects_empty_prompt(model_and_tokenizer):
    model, tok = model_and_tokenizer
    with pytest.raises(ValueError):
        continue_text(model, tok, "", ARGS, DEVICE)


def test_interactive_loop_answers_each_prompt_until_quit(model_and_tokenizer, capsys):
    model, tok = model_and_tokenizer
    interactive_loop(model, tok, ARGS, DEVICE,
                     read=scripted(["ROMEO:", "", "To be", "quit", "never read"]))

    out = capsys.readouterr().out
    # Greedy decoding is deterministic, so each answer can be predicted.
    assert continue_text(model, tok, "ROMEO:", ARGS, DEVICE) in out
    assert continue_text(model, tok, "To be", ARGS, DEVICE) in out
    assert "never read" not in out


def test_interactive_loop_turns_backslash_n_into_newline(model_and_tokenizer, capsys):
    model, tok = model_and_tokenizer
    interactive_loop(model, tok, ARGS, DEVICE, read=scripted([r"ROMEO:\nTo"]))
    assert continue_text(model, tok, "ROMEO:\nTo", ARGS, DEVICE) in capsys.readouterr().out


def test_interactive_loop_stops_cleanly_at_end_of_input(model_and_tokenizer):
    model, tok = model_and_tokenizer
    interactive_loop(model, tok, ARGS, DEVICE, read=scripted([]))  # must not raise


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------

@pytest.fixture
def chat_model_and_tokenizer():
    tok = BPETokenizer()
    tok.train(CHAT_TEXT, num_merges=20)
    torch.manual_seed(0)
    model = MiniLLM(vocab_size=tok.vocab_size, embedding_dim=32, num_heads=4,
                    num_layers=2, max_seq_length=16, dropout=0.0).eval()
    return model, tok


def test_chat_prompt_matches_corpus_layout():
    assert chat_prompt("hello") == (
        f"{config.CHAT_USER_PREFIX} hello\n{config.CHAT_BOT_PREFIX}"
    )


def test_chat_prompt_keeps_the_message_on_one_line():
    assert chat_prompt("  hello\nthere  ") == chat_prompt("hello there")


def test_newline_token_ids_are_exactly_the_tokens_with_a_line_break(chat_model_and_tokenizer):
    _, tok = chat_model_and_tokenizer
    ids = newline_token_ids(tok)
    assert ids  # "\n" (and "\n\n") are in the vocab
    assert all("\n" in tok.itos[i] for i in ids)
    assert all("\n" not in tok.itos[i] for i in set(tok.itos) - ids)


def test_chat_reply_is_the_first_line_of_the_continuation(chat_model_and_tokenizer):
    model, tok = chat_model_and_tokenizer
    reply, new_tokens = chat_reply(model, tok, "Hi, how are you?", CHAT_ARGS, DEVICE)

    # Greedy decoding: the reply is what plain continuation writes after the
    # prompt, up to the first line break.
    prompt = chat_prompt("Hi, how are you?")
    full = continue_text(model, tok, prompt, CHAT_ARGS, DEVICE)
    assert reply == full[len(prompt):].split("\n", 1)[0].strip()
    assert "\n" not in reply

    # Generation stopped right after the first newline token (or ran to the
    # limit if there was none), rather than always using every token.
    new_ids = tok.encode(full)[len(tok.encode(prompt)):]
    stops = [i for i, t in enumerate(new_ids) if t in newline_token_ids(tok)]
    assert new_tokens == (stops[0] + 1 if stops else CHAT_ARGS.max_new_tokens)


def test_chat_reply_handles_a_blank_message(chat_model_and_tokenizer):
    # A blank message still makes a valid prompt ("User: \nBot:").
    model, tok = chat_model_and_tokenizer
    reply, _ = chat_reply(model, tok, "   ", CHAT_ARGS, DEVICE)
    assert isinstance(reply, str)


def test_interactive_chat_prints_bot_replies(chat_model_and_tokenizer, capsys):
    model, tok = chat_model_and_tokenizer
    interactive_loop(model, tok, CHAT_ARGS, DEVICE, chat=True,
                     read=scripted(["Hi, how are you?", "quit"]))

    reply, _ = chat_reply(model, tok, "Hi, how are you?", CHAT_ARGS, DEVICE)
    assert f"{config.CHAT_BOT_PREFIX} {reply}" in capsys.readouterr().out
