"""
Mini LLM — browser interface.

Serves a small web page for prompting the trained model, using only the
Python standard library (no Flask etc. to install):

    checkpoints/  (saved by train.py)
          |
    MiniLLM + BPE tokenizer, loaded once      (checkpoint.py)
          |
    http.server on 127.0.0.1
        GET  /              -> web/index.html
        GET  /api/info      -> checkpoint + default settings as JSON
        POST /api/generate  -> {prompt, settings} in, continuation out
          |
    continue_text()  (generate.py, same code path as the CLI)

Run (after `python train.py`):

    python app.py
    python app.py --port 8080 --checkpoint-dir checkpoints/untied_baseline

then open http://127.0.0.1:8000 in a browser. The server only listens on
this machine (127.0.0.1) unless --host says otherwise.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import torch

import config
from checkpoint import load_checkpoint
from generate import continue_text
from model import get_device

INDEX_HTML = Path(__file__).resolve().parent / "web" / "index.html"

# Upper limits on what one request may ask for, so a typo like
# max_new_tokens=100000 can't tie up the CPU for minutes.
MAX_NEW_TOKENS_LIMIT = 500
MAX_PROMPT_CHARS = 2000


def parse_settings(payload):
    """
    Validate a /api/generate request body and return (prompt, settings).
    Missing settings fall back to config.py's generation defaults.
    Raises ValueError with a message meant for the user.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Type a prompt first.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt is too long (max {MAX_PROMPT_CHARS} characters).")

    def number(key, default, cast, low, high):
        value = payload.get(key)
        if value is None or value == "":
            return default
        try:
            value = cast(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number.")
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        return value

    settings = SimpleNamespace(
        max_new_tokens=number("max_new_tokens", config.MAX_NEW_TOKENS, int,
                              1, MAX_NEW_TOKENS_LIMIT),
        temperature=number("temperature", config.TEMPERATURE, float, 0.0, 2.0),
        top_k=number("top_k", config.TOP_K or 0, int, 0, 100_000),
        seed=number("seed", None, int, 0, 2**32 - 1),
    )
    return prompt, settings


class MiniLLMServer(ThreadingHTTPServer):
    """An HTTP server that holds one loaded model for every request."""

    def __init__(self, address, model, tokenizer, info, device):
        super().__init__(address, RequestHandler)
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.device = device
        # One generation at a time: requests run on separate threads, but
        # they share one model and torch's global random seed.
        self.lock = threading.Lock()

    def generate(self, prompt, settings):
        with self.lock:
            if settings.seed is not None:
                torch.manual_seed(settings.seed)
            start = time.perf_counter()
            text = continue_text(self.model, self.tokenizer, prompt,
                                 settings, self.device)
            elapsed = time.perf_counter() - start

        # The decoded prompt can differ from what was typed (characters the
        # tokenizer doesn't know become <unk>), so split on the decoded form.
        shown_prompt = self.tokenizer.decode(self.tokenizer.encode(prompt))
        return {
            "prompt": shown_prompt,
            "continuation": text[len(shown_prompt):],
            "new_tokens": settings.max_new_tokens,
            "seconds": round(elapsed, 3),
        }


class RequestHandler(BaseHTTPRequestHandler):
    server: MiniLLMServer

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_body(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            model = self.server.model
            self.send_json(200, {
                "epoch": self.server.info["epoch"],
                "val_loss": self.server.info["val_loss"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "vocab_size": self.server.tokenizer.vocab_size,
                "context_length": model.max_seq_length,
                "device": str(self.server.device),
                "defaults": {
                    "max_new_tokens": config.MAX_NEW_TOKENS,
                    "temperature": config.TEMPERATURE,
                    "top_k": config.TOP_K or 0,
                },
                "limits": {
                    "max_new_tokens": MAX_NEW_TOKENS_LIMIT,
                    "prompt_chars": MAX_PROMPT_CHARS,
                },
            })
        else:
            self.send_json(404, {"error": "Not found."})

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            prompt, settings = parse_settings(payload)
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Request body is not valid JSON."})
            return
        except ValueError as err:
            self.send_json(400, {"error": str(err)})
            return

        try:
            self.send_json(200, self.server.generate(prompt, settings))
        except ValueError as err:  # e.g. a prompt that encodes to no tokens
            self.send_json(400, {"error": str(err)})

    def send_json(self, status, data):
        self.send_body(status, json.dumps(data).encode("utf-8"),
                       "application/json; charset=utf-8")

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Only log generations, not every page/info request.
        if self.command == "POST":
            super().log_message(fmt, *args)


def parse_args():
    parser = argparse.ArgumentParser(description="Browser interface for a trained MiniLLM.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="defaults to config.CHECKPOINT_DIR")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, tokenizer, info = load_checkpoint(args.checkpoint_dir, map_location=device)

    server = MiniLLMServer((args.host, args.port), model, tokenizer, info, device)
    print(f"Loaded checkpoint from epoch {info['epoch']} "
          f"(val loss {info['val_loss']:.4f}) on {device}")
    print(f"Open http://{args.host}:{server.server_port} in your browser "
          f"(Ctrl+C to stop).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
