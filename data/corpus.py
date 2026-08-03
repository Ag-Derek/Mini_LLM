"""
data/corpus.py
===============

Single place that knows how to load the project's training corpus.

Why this exists:
    Before this, loading the corpus meant `with open(...)` scattered
    inside whichever script needed text (bpe_tokenizer.py's __main__,
    dataset.py's __main__, eventually train.py). Every one of those
    would need to know the exact file path, encoding, and how to handle
    a missing file. Centralizing it here means the rest of the project
    just does:

        from data.corpus import load_corpus
        text = load_corpus()

    and never touches a file path directly.

This module deliberately returns RAW text and nothing else -- no
lowercasing, no tokenizing, no train/val splitting. Those decisions
belong to whichever tokenizer or dataset script consumes the text, not
to the loader. Keeping this dumb means it stays reusable no matter which
tokenizer (char / word / BPE) or corpus (Tiny Shakespeare today, maybe
something else later) is in play.
"""

from pathlib import Path

import config


def _default_corpus_path() -> Path:
    """
    Resolve the corpus path relative to the project root (the parent of
    this data/ folder), not the current working directory -- so
    load_corpus() works the same whether a script is run from the
    project root, from inside data/, or imported from elsewhere.
    """
    project_root = Path(__file__).resolve().parent.parent
    return project_root / config.DATA_DIR / config.CORPUS_FILENAME


def load_corpus(path: str | Path | None = None) -> str:
    """
    Load and return the raw training corpus as a single string.

    path: optional override. Defaults to config.DATA_DIR /
          config.CORPUS_FILENAME (currently data/tiny_corpus.txt).

    Raises FileNotFoundError with a clear message if the corpus file is
    missing, rather than letting a bare `open()` traceback surface
    somewhere deep in a tokenizer or dataset script.
    """
    corpus_path = Path(path) if path is not None else _default_corpus_path()

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found at {corpus_path}. Expected "
            f"config.CORPUS_FILENAME ('{config.CORPUS_FILENAME}') inside "
            f"config.DATA_DIR ('{config.DATA_DIR}')."
        )

    return corpus_path.read_text(encoding="utf-8")


def train_val_split(
    text: str, val_ratio: float = config.VAL_RATIO
) -> tuple[str, str]:
    """
    Split raw text into a (train_text, val_text) pair.

    The split is a single contiguous cut -- the last `val_ratio` fraction
    of characters becomes the validation set, everything before it is
    training data. This is deliberately NOT a random/shuffled split:
    shuffling individual lines or windows would let the tokenizer/model
    train on text that sits chronologically after (and was learned
    "from") material in the validation set, undermining the point of
    holding data out. A single contiguous split is the standard approach
    for this kind of small, single-document language modeling corpus.

    val_ratio defaults to config.VAL_RATIO (0.1 -- i.e. a 90/10 split).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")

    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


if __name__ == "__main__":
    # Quick manual sanity check: python -m data.corpus
    text = load_corpus()
    print(f"Corpus path: {_default_corpus_path()}")
    print(f"Corpus length: {len(text):,} characters")
    print(f"First 200 characters:\n{text[:200]!r}")

    train_text, val_text = train_val_split(text)
    print(f"\nTrain/val split (val_ratio={config.VAL_RATIO}):")
    print(f"  train: {len(train_text):,} characters")
    print(f"  val:   {len(val_text):,} characters")
