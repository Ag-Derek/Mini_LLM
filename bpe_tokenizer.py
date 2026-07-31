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

Like WordTokenizer, decode(encode(text)) == text is NOT guaranteed:
lowercasing and whitespace normalization mean we can only reconstruct a
readable version of the text, not the byte-exact original. What we do
verify is the token-level round trip: encode(decode(encode(text))) ==
encode(text) -- i.e. re-encoding the reconstructed text gives back the
same token IDs.

The sanity check confirms:

    - merge learning works (vocab actually grows subword tokens)
    - encoding applies learned merges in the right order
    - decoding works
    - token-level round trip holds
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
    Byte-Pair-Encoding tokenizer (character-level BPE, word-bounded).

    Training:
        1. Split text into "words" using the same word/punctuation
           pattern as WordTokenizer (so punctuation never merges across
           a word boundary).
        2. Represent each word as a list of characters plus an
           end-of-word marker (EOW), e.g. "cat" -> ["c", "a", "t", "</w>"].
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

    # One or more word chars, OR a single non-space/non-word char.
    # Same pattern as WordTokenizer so punctuation is always its own unit.
    _WORD_PATTERN = re.compile(r"\w+|[^\w\s]")

    # Marks the end of a word so merges never cross word boundaries and
    # so decode() can tell where one word ends and the next begins.
    EOW = "</w>"

    def __init__(self):
        self.stoi: dict[str, int] = {}                 # string -> int
        self.itos: dict[int, str] = {}                 # int -> string
        self.merges: dict[tuple[str, str], int] = {}   # pair -> rank (learned order)

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
        """
        text = text.lower()
        words = self._WORD_PATTERN.findall(text)

        word_freqs = Counter(words)

        # Each distinct word starts as a list of its characters + EOW.
        splits: dict[str, list[str]] = {
            word: list(word) + [self.EOW] for word in word_freqs
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
        # learned. Seeding with base characters + EOW up front, then
        # adding each merge's output the moment it's created, means every
        # symbol that was ever a valid subword stays in the vocab.
        base_chars = {ch for word in word_freqs for ch in word}
        all_symbols: set[str] = set(base_chars) | {self.EOW}

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
        Convert text into token IDs. Text is lowercased and split into
        word/punctuation units (same pattern as WordTokenizer), each unit
        is broken into characters + EOW, learned merges are replayed, and
        the resulting subword symbols are looked up. A symbol never seen
        during training (i.e. not produced by any merge and not a base
        character in the vocab) maps to <unk>.
        """
        text = text.lower()
        words = self._WORD_PATTERN.findall(text)

        ids: list[int] = []
        unk_id = self.stoi[config.UNK_TOKEN]
        for word in words:
            symbols = self._apply_merges(list(word) + [self.EOW])
            for symbol in symbols:
                ids.append(self.stoi.get(symbol, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Join subword tokens back into a readable string. Symbols are
        stitched together within a word using the EOW marker to find
        word boundaries, then words are joined the same way
        WordTokenizer does: punctuation glued to the previous word, a
        space everywhere else. Best-effort reconstruction, not the exact
        original text/whitespace.
        """
        no_space_before = set(".,!?;:')]}\"'")

        tokens = [self.itos[i] for i in ids]

        words: list[str] = []
        buf = ""
        for tok in tokens:
            if tok.endswith(self.EOW):
                buf += tok[: -len(self.EOW)]
                words.append(buf)
                buf = ""
            else:
                buf += tok
        if buf:
            words.append(buf)

        pieces: list[str] = []
        for w in words:
            glue = len(w) == 1 and w in no_space_before
            if pieces and not glue:
                pieces.append(" ")
            pieces.append(w)

        return "".join(pieces)

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
    # Quick manual sanity check: python bpe_tokenizer.py
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

    # Like WordTokenizer, exact string round-trip isn't guaranteed (we
    # lowercase and normalize whitespace), so check the token-level
    # round trip instead: re-encoding the decoded text should give back
    # the same IDs.
    re_ids = tok.encode(decoded)
    assert re_ids == ids, "Token-level round-trip failed!"
    print("Token round-trip OK.")

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
if __name__ == "__main__":

    # your existing tests above...

    print("\nTop learned merges:")

    for i, (pair, rank) in enumerate(
        sorted(tok.merges.items(), key=lambda x: x[1]),
        start=1
    ):
        print(
            f"{i:2}. {pair[0]} + {pair[1]} -> {pair[0] + pair[1]}"
        )