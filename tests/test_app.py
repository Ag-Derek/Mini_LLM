"""
app.py tests: request validation, and the real HTTP server end to end
with a tiny untrained model (no checkpoint needed).
"""

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

import config
from app import MAX_NEW_TOKENS_LIMIT, MiniLLMServer, parse_settings
from bpe_tokenizer import BPETokenizer
from model import MiniLLM

TEXT = "ROMEO:\nTo be, or not to be, that is the question:\n" * 20


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def test_parse_settings_fills_in_defaults():
    prompt, s = parse_settings({"prompt": "ROMEO:"})
    assert prompt == "ROMEO:"
    assert s.max_new_tokens == config.MAX_NEW_TOKENS
    assert s.temperature == config.TEMPERATURE
    assert s.seed is None


def test_parse_settings_reads_values():
    _, s = parse_settings({"prompt": "x", "max_new_tokens": "20",
                           "temperature": 0, "top_k": 0, "seed": 7})
    assert (s.max_new_tokens, s.temperature, s.top_k, s.seed) == (20, 0.0, 0, 7)


@pytest.mark.parametrize("payload", [
    {"prompt": ""},
    {"prompt": "   "},
    {"prompt": 5},
    {"prompt": "x" * 5000},
    {"prompt": "x", "max_new_tokens": MAX_NEW_TOKENS_LIMIT + 1},
    {"prompt": "x", "max_new_tokens": 0},
    {"prompt": "x", "temperature": -1},
    {"prompt": "x", "temperature": "hot"},
    ["not", "an", "object"],
])
def test_parse_settings_rejects_bad_input(payload):
    with pytest.raises(ValueError):
        parse_settings(payload)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    tok = BPETokenizer()
    tok.train(TEXT, num_merges=20)
    torch.manual_seed(0)
    model = MiniLLM(vocab_size=tok.vocab_size, embedding_dim=32, num_heads=4,
                    num_layers=2, max_seq_length=16, dropout=0.0).eval()
    info = {"epoch": 1, "val_loss": 2.5}

    srv = MiniLLMServer(("127.0.0.1", 0), model, tok, info, torch.device("cpu"))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(server, path, payload=None):
    url = f"http://127.0.0.1:{server.server_port}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_index_page_is_served(server):
    status, body = call(server, "/")
    assert status == 200
    assert b"<title>Mini LLM</title>" in body


def test_info_reports_checkpoint(server):
    status, body = call(server, "/api/info")
    info = json.loads(body)
    assert status == 200
    assert info["epoch"] == 1 and info["val_loss"] == 2.5
    assert info["parameters"] == sum(p.numel() for p in server.model.parameters())


def test_generate_returns_prompt_and_continuation(server):
    status, body = call(server, "/api/generate",
                        {"prompt": "ROMEO:", "max_new_tokens": 6, "temperature": 0})
    data = json.loads(body)
    assert status == 200
    assert data["prompt"] == "ROMEO:"
    assert data["continuation"]
    assert data["new_tokens"] == 6


def test_same_seed_gives_same_text(server):
    payload = {"prompt": "To be", "max_new_tokens": 8, "temperature": 1.0, "seed": 3}
    first = json.loads(call(server, "/api/generate", payload)[1])
    second = json.loads(call(server, "/api/generate", payload)[1])
    assert first["continuation"] == second["continuation"]


def test_generate_rejects_bad_requests(server):
    status, body = call(server, "/api/generate", {"prompt": ""})
    assert status == 400 and "error" in json.loads(body)
    assert call(server, "/api/nope")[0] == 404
