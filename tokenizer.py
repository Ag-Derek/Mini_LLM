"""
tokenizer.py

Tokenizers for mini_llm. We start with a character-level tokenizer
(simplest possible) and will add WordTokenizer and BPETokenizer later.

All tokenizers share the same interface (see Tokenizer base class) so the
rest of the pipeline (dataset.py, train.py, generate.py) never needs to
know which one is active.

===========================================================
HOW TO RUN THIS TEST
===========================================================

1. Install Python
-----------------
Make sure Python is installed.

Check installation:

    python --version

Expected output example:

    Python 3.x.x


If Python is not found, install it from:
https://www.python.org/downloads/


2. Open PowerShell
------------------
Navigate to the folder containing this file.

Example:

    cd C:\Users\HomePC\Documents\Python


3. Verify the file exists
-------------------------
Check that tokenizer.py is in the current directory:

    dir

Expected output:

    tokenizer.py


4. Run the tokenizer
--------------------
Execute:

    python tokenizer.py


If Python is not added to PATH, use the full Python path:

    & "C:\Path\To\Python\python.exe" tokenizer.py


5. Expected output
------------------
A successful run should display:

    - vocabulary size
    - vocabulary dictionary
    - encoded token IDs
    - decoded text
    - unknown token handling test
    - Round-trip OK.


The sanity check verifies:

    Text
      |
      v
    Tokenizer
      |
      v
    Token IDs
      |
      v
    Decoded Text

The decoded text should match the original input.

===========================================================
"""

import json
from pathlib import Path


class Tokenizer:
    """Base interface every tokenizer must implement."""

    def train(self, text: str) -> None:
        """Build the vocabulary from raw text."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token ids."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token ids back into a string."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    """
    Simplest possible tokenizer: one token per character.

    Vocab is just the sorted set of unique characters seen in training text.
    No unknown-token handling beyond what you train on — if you encode text
    with a character that wasn't in the training data, it will raise a
    KeyError. That's intentional for now: it keeps the code honest about
    what "training" a tokenizer actually means, and we'll handle unknowns
    (<unk>) properly when we get to the word tokenizer.
    """

    def __init__(self):
        self.stoi: dict[str, int] = {}   # string -> int
        self.itos: dict[int, str] = {}   # int -> string

    def train(self, text: str) -> None:

        chars = sorted(set(text))

        chars = ["<unk>"] + chars

        self.stoi = {
        ch: i 
        for i, ch in enumerate(chars)
            }

        self.itos = {
        i: ch
        for i, ch in enumerate(chars)
            }

    def encode(self, text: str) -> list[int]:
        return [
        self.stoi.get(ch, self.stoi["<unk>"])
        for ch in text
    ]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(i): ch for ch, i in tok.stoi.items()}
        return tok


if __name__ == "__main__":
    # Quick manual sanity check: python tokenizer.py
    sample = """
    hello world, this is a tiny llm tokenizer test!
    machine learning is amazing.
    artificial intelligence is changing the world.
    deep learning uses neural networks.
    1234567890
    """

    tok = CharTokenizer()
    tok.train(sample)

    print(f"vocab_size: {tok.vocab_size}")
    print(f"vocab: {tok.stoi}")

    ids = tok.encode(sample)
    print(f"encoded: {ids}")

    decoded = tok.decode(ids)
    print(f"decoded: {decoded}")
    numbers = "12345"

    ids = tok.encode(numbers)

    print(ids)
    assert decoded == sample, "Round-trip failed!"
    print("Round-trip OK.")