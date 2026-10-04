"""
bpe_tokenizer.py

Byte-Pair-Encoding (BPE) tokenizer for mini_llm.

Sits alongside tokenizer.py (Tokenizer base interface + CharTokenizer) and
word_tokenizer.py (WordTokenizer), and imports the base interface from
tokenizer.py, so every tokenizer in the project shares one contract and
the rest of the pipeline (dataset.py, train.py, generate.py) never needs
to know which one is active.

WHY BPE, AFTER CHAR AND WORD
-----------------------------
CharTokenizer: tiny vocab, but every word is many tokens.
WordTokenizer: short sequences, but the vocab explodes and anything not
seen during training becomes a single <unk> -- the whole word is lost.

BPE is the middle ground: it starts from individual characters (so it can
always fall back to spelling an unseen word out letter by letter) and
greedily merges the most frequent adjacent pair into a new subword token,
repeated `num_merges` times. Common words collapse into one token, rare
words fall back to a handful of subword pieces, and *unseen* words are
only partially unknown (unknown characters map to <unk>, everything else
still encodes).


===========================================================
HOW TO RUN THIS TEST
===========================================================

Run it the same way as tokenizer.py / word_tokenizer.py, just point at
this file instead:

    python bpe_tokenizer.py

(bpe_tokenizer.py must live in the same folder as tokenizer.py, since it
imports Tokenizer from there.)


===========================================================
WHAT THE SANITY CHECK VERIFIES
===========================================================

    Raw Text
        |
        v
    BPETokenizer
        |
        v
    Token IDs
        |
        v
    Decoder
        |
        v
    Reconstructed Text

Unlike WordTokenizer, whitespace (spaces AND newlines) is kept as part of
the tokens, so decode(encode(text)) == text exactly, as long as every
character in `text` was seen during training. For Tiny Shakespeare this
matters: line breaks and blank lines between speakers are real structure
the model should learn to produce.

The sanity check confirms:

    - merge learning works (vocab actually grows subword tokens)
    - encoding applies learned merges in the right order
    - exact text round trip holds (whitespace and newlines included)
    - unseen characters are handled correctly (fall back to <unk>)

===========================================================
"""

import json
import re
from collections import Counter
from pathlib import Path

import config
from tokenizer import Tokenizer


class BPETokenizer(Tokenizer):
    """
    Byte-Pair-Encoding tokenizer (character-level BPE, chunk-bounded).

    Training:
        1. Split text into chunks GPT-2 style: a word or punctuation run
           together with its leading space (" cat", " ,"), or a run of
           whitespace such as "\n\n". Merges never cross a chunk
           boundary, so a token can't straddle two words.
        2. Represent each chunk as a list of characters,
           e.g. " cat" -> [" ", "c", "a", "t"].
        3. Repeatedly find the most frequent adjacent symbol pair across
           the whole corpus and merge it into a single new symbol.
           Each merge is recorded, in order, in self.merges.
        4. The final vocab is every symbol that appears anywhere in the
           corpus after all merges are applied, plus special tokens.

    Encoding a new word replays the learned merges in the order they were
    learned (lowest rank first) until no more apply -- this is standard
    BPE encoding.

    Unknown symbols (individual characters never seen during training)
    map to <unk>. Because encoding falls back to raw characters, only
    genuinely unseen characters are lost -- unseen *words* built from
    known characters still encode, usually as multiple subword tokens.
    """

    # Same idea as GPT-2's pre-tokenizer, in order of preference:
    #   " ?\w+"         a word, with its leading space if it has one
    #   " ?[^\w\s]+"    a run of punctuation, with its leading space
    #   "\s+(?!\S)"     whitespace, leaving the last space for the next word
    #   "\s+"           any remaining whitespace (e.g. a trailing newline)
    # Every character of the input lands in exactly one chunk, which is
    # what makes decode() an exact inverse of encode().
    _CHUNK_PATTERN = re.compile(r" ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+")

    def __init__(self):
        self.stoi: dict[str, int] = {}                 # string -> int
        self.itos: dict[int, str] = {}                 # int -> string
        self.merges: dict[tuple[str, str], int] = {}   # pair -> rank (learned order)
        # chunk -> token ids. The same chunks (" the", "\n") repeat
        # constantly, so replaying merges once per distinct chunk instead
        # of once per occurrence makes encoding a large corpus far faster.
        self._encode_cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------
    # training
    # ---------------------------------------------------------------

    def train(self, text: str, num_merges: int = config.NUM_MERGES) -> None:
        """
        Build the vocabulary from raw text by learning up to `num_merges`
        BPE merges. Training stops early if there are no more repeated
        adjacent pairs left to merge.

        num_merges defaults to config.NUM_MERGES so every part of the
        project learns the same-sized vocab unless a caller deliberately
        overrides it (e.g. a quick demo with a smaller corpus).

        Case is preserved deliberately (no .lower() here): for a corpus
        like Tiny Shakespeare, capitalization carries real signal --
        character names (ROMEO, JULIET) and sentence starts -- that a
        language model can learn from. The tradeoff is a larger vocab,
        since "the" and "The" now train as distinct symbol sequences.
        """
        words = self._CHUNK_PATTERN.findall(text)

        word_freqs = Counter(words)

        # Each distinct chunk starts as a list of its characters.
        splits: dict[str, list[str]] = {
            word: list(word) for word in word_freqs
        }

        # Track every symbol that ever exists during training, not just
        # whatever happens to survive in `splits` after the final merge.
        # Bug this fixes: if, say, every occurrence of "th" in the corpus
        # is later merged into "the", then "th" never appears in the final
        # splits and would be dropped from the vocab entirely -- even
        # though the tokenizer clearly learned it as an intermediate
        # merge. Encoding an unseen word like "throw" would then produce
        # "th" via _apply_merges but find it missing from stoi and fall
        # back to <unk>, silently wasting a merge the model actually
        # learned. Seeding with base characters up front, then adding each
        # merge's output the moment it's created, means every symbol that
        # was ever a valid subword stays in the vocab.
        base_chars = {ch for word in word_freqs for ch in word}
        all_symbols: set[str] = set(base_chars)

        merges_in_order: list[tuple[str, str]] = []

        for _ in range(num_merges):
            pair_freq = self._count_pairs(word_freqs, splits)

            if not pair_freq:
                break  # nothing left that repeats

            best_pair, _ = pair_freq.most_common(1)[0]
            merged_token = best_pair[0] + best_pair[1]

            for word in splits:
                splits[word] = self._merge_pair(splits[word], best_pair, merged_token)

            merges_in_order.append(best_pair)
            all_symbols.add(merged_token)

        self.merges = {pair: rank for rank, pair in enumerate(merges_in_order)}

        vocab = config.SPECIAL_TOKENS + sorted(all_symbols)

        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for i, tok in enumerate(vocab)}
        self._encode_cache = {}

    @staticmethod
    def _count_pairs(
        word_freqs: dict[str, int], splits: dict[str, list[str]]
    ) -> Counter:
        """Count frequency of every adjacent symbol pair across the corpus."""
        pair_freq: Counter = Counter()
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for a, b in zip(symbols, symbols[1:]):
                pair_freq[(a, b)] += freq
        return pair_freq

    @staticmethod
    def _merge_pair(
        symbols: list[str], pair: tuple[str, str], merged: str
    ) -> list[str]:
        """Replace every adjacent occurrence of `pair` in `symbols` with `merged`."""
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """Repeatedly apply the lowest-rank applicable merge until none apply."""
        symbols = list(symbols)
        while True:
            candidates = [
                (self.merges[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.merges
            ]
            if not candidates:
                break
            _, best_pair = min(candidates)
            merged_token = best_pair[0] + best_pair[1]
            symbols = self._merge_pair(symbols, best_pair, merged_token)
        return symbols

    # ---------------------------------------------------------------
    # encode / decode
    # ---------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs. Text is split into chunks with the
        same pattern as train() (words keep their leading space,
        whitespace runs are their own chunks), each chunk is broken into
        characters, learned merges are replayed, and the resulting
        subword symbols are looked up. A symbol never seen during
        training (i.e. not produced by any merge and not a base character
        in the vocab) maps to <unk>.

        Case is preserved to match train() -- see its docstring. This
        means a word's case must match training for merges to apply the
        same way: "Romeo" and "ROMEO" are different symbol sequences
        unless both appeared during training.
        """
        ids: list[int] = []
        unk_id = self.stoi[config.UNK_TOKEN]
        for chunk in self._CHUNK_PATTERN.findall(text):
            chunk_ids = self._encode_cache.get(chunk)
            if chunk_ids is None:
                symbols = self._apply_merges(list(chunk))
                chunk_ids = [self.stoi.get(symbol, unk_id) for symbol in symbols]
                self._encode_cache[chunk] = chunk_ids
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Join subword tokens back into a string. Every token already
        carries its own whitespace (" the", "\\n"), so this is a plain
        concatenation, and decode(encode(text)) == text whenever every
        character in `text` was seen during training.
        """
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    # ---------------------------------------------------------------
    # persistence
    # ---------------------------------------------------------------

    def save(self, path: str) -> None:
        # Merge ranks must round-trip through JSON in learned order, so
        # store them as an ordered list of [a, b] pairs rather than a
        # dict (JSON keys can't be tuples).
        ordered_merges = [
            list(pair) for pair, _ in sorted(self.merges.items(), key=lambda kv: kv[1])
        ]
        Path(path).write_text(
            json.dumps(
                {"stoi": self.stoi, "merges": ordered_merges},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(i): w for w, i in tok.stoi.items()}
        tok.merges = {tuple(pair): rank for rank, pair in enumerate(data["merges"])}
        return tok


if __name__ == "__main__":
    # Quick manual sanity check on a tiny demo string: python bpe_tokenizer.py
    sample = """
    hello world, this is a tiny llm tokenizer test!
    machine learning is amazing.
    artificial intelligence is changing the world.
    deep learning uses neural networks.
    1234567890
    """

    tok = BPETokenizer()
    tok.train(sample, num_merges=40)

    print(f"vocab_size: {tok.vocab_size}")
    print(f"vocab: {tok.stoi}")
    print(f"num merges learned: {len(tok.merges)}")

    ids = tok.encode(sample)
    print(f"encoded: {ids}")

    decoded = tok.decode(ids)
    print(f"decoded: {decoded}")

    # Whitespace (including newlines and indentation) is part of the
    # tokens, so decoding must give back the exact original string.
    assert decoded == sample, "Exact round-trip failed!"
    print("Exact round-trip OK.")

    # Test unseen-word handling: these words never appeared in training,
    # but they're built entirely from known characters, so BPE can still
    # spell them out (unlike WordTokenizer, which would collapse the
    # whole word to a single <unk>).
    unseen_word = "outstanding"
    ids = tok.encode(unseen_word)
    print(ids)
    print(tok.decode(ids))

    # Test genuinely unknown character handling: a character that never
    # appeared in training has no base entry in the vocab at all, so it
    # (and only it) should fall back to <unk>.
    unseen_chars = "hello world é"
    ids = tok.encode(unseen_chars)
    print(ids)
    print(tok.decode(ids))

    print("\nTop learned merges (tiny demo string):")
    for i, (pair, rank) in enumerate(
        sorted(tok.merges.items(), key=lambda x: x[1]),
        start=1
    ):
        print(f"{i:2}. {pair[0]} + {pair[1]} -> {pair[0] + pair[1]}")

    # ------------------------------------------------------------------
    # Real-corpus comparison: train a second tokenizer on the full Tiny
    # Shakespeare corpus (config.NUM_MERGES = 500 merges) and compare
    # against the tiny 40-merge demo above. This is the number that
    # actually matters for the project -- the demo above just exists to
    # sanity-check the mechanics on a string small enough to eyeball.
    # ------------------------------------------------------------------
    from data.corpus import load_corpus

    print("\n--- Training on the real corpus (Tiny Shakespeare) ---")
    corpus_text = load_corpus()
    print(f"Corpus length: {len(corpus_text):,} characters")

    corpus_tok = BPETokenizer()
    corpus_tok.train(corpus_text)  # num_merges defaults to config.NUM_MERGES

    print(f"\n{'':20}{'demo string':>15}{'real corpus':>15}")
    print(f"{'vocab_size':20}{tok.vocab_size:>15}{corpus_tok.vocab_size:>15}")
    print(f"{'merges learned':20}{len(tok.merges):>15}{len(corpus_tok.merges):>15}")

    print("\nTop 15 merges learned on the real corpus:")
    for i, (pair, rank) in enumerate(
        sorted(corpus_tok.merges.items(), key=lambda x: x[1])[:15],
        start=1
    ):
        print(f"{i:2}. {pair[0]!r} + {pair[1]!r} -> {pair[0] + pair[1]!r}")