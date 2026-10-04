"""
generate.py tests: one-shot continuation and the interactive prompt loop,
using a tiny untrained model so no checkpoint is needed.
"""

from types import SimpleNamespace

import pytest
import torch

from bpe_tokenizer import BPETokenizer
from generate import continue_text, interactive_loop
from model import MiniLLM

TEXT = "ROMEO:\nTo be, or not to be, that is the question:\n" * 20
ARGS = SimpleNamespace(max_new_tokens=5, temperature=0, top_k=0)
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
