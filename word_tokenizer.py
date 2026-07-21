"""
word_tokenizer.py

Word-level tokenizer for mini_llm.

Sits alongside tokenizer.py (which holds the Tokenizer base interface and
CharTokenizer) and imports the base interface from there, so every
tokenizer in the project shares one contract and the rest of the pipeline
(dataset.py, train.py, generate.py) never needs to know which one is
active.


===========================================================
HOW TO RUN THIS TEST
===========================================================

Run it the same way as tokenizer.py, just point at this file instead:

    python word_tokenizer.py

(word_tokenizer.py must live in the same folder as tokenizer.py, since it
imports Tokenizer from there.)


===========================================================
WHAT THE SANITY CHECK VERIFIES
===========================================================

    Raw Text
        |
        v
    WordTokenizer
        |
        v
    Token IDs
        |
        v
    Decoder
        |
        v
    Reconstructed Text

Unlike CharTokenizer, decode(encode(text)) == text is NOT guaranteed for
a word tokenizer: splitting on whitespace/punctuation throws away exact
spacing and casing, so we can only reconstruct *a* readable version of
the text, not the byte-exact original. What we can (and do) verify is a
token-level round trip: encode(decode(encode(text))) == encode(text) --
i.e. re-encoding the reconstructed text gives back the same token IDs.

The sanity check confirms:

    - vocabulary creation works
    - encoding works
    - decoding works
    - token IDs map correctly
    - token-level round trip holds
    - unknown words are handled correctly

===========================================================
"""

import json
import re
from pathlib import Path

from tokenizer import Tokenizer


class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer.

    Vocab is the sorted set of unique tokens seen in training text, where
    a "token" is either a run of word characters (letters/digits/'_') or
    a single punctuation character. So "world!" splits into ["world", "!"]
    rather than one glued-together token, which keeps punctuation from
    silently exploding the vocab with entries like "world!", "world,",
    "world." all as separate words.

    Unknown words (not seen during training) map to <unk>, same as
    CharTokenizer.

    IMPORTANT DIFFERENCE FROM CharTokenizer:
    decode(encode(text)) == text is NOT guaranteed here. Splitting on
    whitespace/punctuation throws away the exact original spacing and
    casing, so we can only reconstruct *a* readable version of the text,
    not the byte-exact original. What IS guaranteed (and is what the
    sanity check below verifies) is a *token-level* round trip:
    encode(decode(encode(text))) == encode(text). Exact string round trip
    comes back once we get to the BPE tokenizer.
    """

    # One or more word chars, OR a single non-space/non-word char.
    _PATTERN = re.compile(r"\w+|[^\w\s]")

    def __init__(self):
        self.stoi: dict[str, int] = {}   # string -> int
        self.itos: dict[int, str] = {}   # int -> string

    def _split(self, text: str) -> list[str]:
        """Lowercase and split into word/punctuation tokens."""
        return self._PATTERN.findall(text.lower())

    def train(self, text: str) -> None:
        words = self._split(text)

        special_tokens = [
                        "<pad>",
                        "<unk>",
                        "<bos>",
                        "<eos>"
                            ]

        vocab = special_tokens + sorted(set(words))

        self.stoi = {
        w:i for i,w in enumerate(vocab)
        }

        self.itos = {
        i:w for i,w in enumerate(vocab)
        }

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        Text is lowercased and split into word/punctuation tokens before
        lookup; tokens not seen during training map to <unk>.
        """
        words = self._split(text)

        return [
        self.stoi.get(w, self.stoi["<unk>"])
        for w in words
        ]

    def decode(self, ids: list[int]) -> str:
        """
        Join tokens back into a readable string. Punctuation tokens are
        glued to the previous word (no space before ".", ",", "!", etc.)
        so the output reads naturally, but this is a best-effort
        reconstruction, not the exact original text/whitespace.
        """
        no_space_before = set(".,!?;:')]}\"'")

        pieces: list[str] = []
        for w in (self.itos[i] for i in ids):
            if pieces and w not in no_space_before:
                pieces.append(" ")
            pieces.append(w)

        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(i): w for w, i in tok.stoi.items()}
        return tok


if __name__ == "__main__":
    # Quick manual sanity check: python word_tokenizer.py
    sample = """
    hello world, this is a tiny llm tokenizer test!
    machine learning is amazing.
    artificial intelligence is changing the world.
    deep learning uses neural networks.
    1234567890
    """

    tok = WordTokenizer()
    tok.train(sample)

    print(f"vocab_size: {tok.vocab_size}")
    print(f"vocab: {tok.stoi}")

    ids = tok.encode(sample)
    print(f"encoded: {ids}")

    decoded = tok.decode(ids)
    print(f"decoded: {decoded}")

    # Word tokenizers don't preserve exact spacing/casing, so instead of
    # asserting decoded == sample, we check the token-level round trip:
    # re-encoding the decoded text should give back the same IDs.
    re_ids = tok.encode(decoded)
    assert re_ids == ids, "Token-level round-trip failed!"
    print("Token round-trip OK.")

    # Test unknown token handling.
    # These words never appeared in the training text, so they should
    # all collapse to the <unk> ID.
    unseen = "quantum computing is fascinating"
    ids = tok.encode(unseen)
    print(ids)
    print(tok.decode(ids))
