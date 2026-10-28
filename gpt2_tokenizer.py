"""
gpt2_tokenizer.py

GPT-2's own tokenizer, behind the same Tokenizer interface as
tokenizer.py / word_tokenizer.py / bpe_tokenizer.py, so dataset.py,
generate.py and app.py work with it unchanged.

WHY A FOURTH TOKENIZER
----------------------
Pretrained weights only mean something with the vocabulary they were
trained on: GPT-2's token 464 is "The", and its embedding row 464 was
learned for "The". Our own BPETokenizer would assign 464 to something
else entirely, so loading GPT-2 (model.py's from_pretrained_gpt2) needs
GPT-2's exact tokenizer.

It is still BPE -- the same algorithm as bpe_tokenizer.py, with 50,000
merges learned by OpenAI on ~40 GB of web text -- with one difference:
it merges *bytes*, not characters. Every string is UTF-8 bytes, and all
256 bytes are in the base vocab, so there is no <unk>: any text at all,
emoji and accents included, round-trips exactly.

The encoding itself comes from OpenAI's `tiktoken` package (a fast Rust
implementation), which downloads the merge table on first use and caches
it. There is nothing to train, so train() is not supported, and save()
only records which encoding to load.

Special tokens: GPT-2 has exactly one, <|endoftext|> (id 50256), which
separated documents during its training. It is recognized when it
appears in the text, so a corpus can mark the end of a document or chat
turn with it.

    python gpt2_tokenizer.py
"""

import json
from pathlib import Path

from tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"


class GPT2Tokenizer(Tokenizer):
    """GPT-2's byte-level BPE (via tiktoken)."""

    TYPE = "tiktoken"  # the "type" field save() writes; see checkpoint.load_tokenizer

    def __init__(self, encoding_name="gpt2"):
        import tiktoken

        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self.eot_id = self._enc.eot_token

    def train(self, text: str) -> None:
        raise NotImplementedError(
            "GPT2Tokenizer is pretrained -- its vocab must match GPT-2's "
            "weights, so it can't be retrained."
        )

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={END_OF_TEXT})

    def decode(self, ids: list[int]) -> str:
        # A single token can be part of a multi-byte character (e.g. half
        # an emoji); decoding it alone gives U+FFFD instead of raising.
        return self._enc.decode(ids, errors="replace")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"type": self.TYPE, "encoding": self.encoding_name}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "GPT2Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["encoding"])


if __name__ == "__main__":
    # Quick manual sanity check: python gpt2_tokenizer.py
    tok = GPT2Tokenizer()
    print(f"vocab_size: {tok.vocab_size}")

    sample = "ROMEO:\nBut, soft! what light through yonder window breaks?  café 🙂"
    ids = tok.encode(sample)
    print(f"encoded: {ids}")
    print(f"pieces:  {[tok.decode([i]) for i in ids]}")

    # Byte-level BPE: exact round trip for any text, no <unk>.
    assert tok.decode(ids) == sample, "Exact round-trip failed!"
    print("Exact round-trip OK.")

    assert tok.encode("Hi" + END_OF_TEXT) == tok.encode("Hi") + [tok.eot_id]
    print(f"{END_OF_TEXT} -> id {tok.eot_id}")
