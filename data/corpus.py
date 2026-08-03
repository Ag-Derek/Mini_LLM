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


if __name__ == "__main__":
    # Quick manual sanity check: python data/corpus.py
    text = load_corpus()
    print(f"Corpus path: {_default_corpus_path()}")
    print(f"Corpus length: {len(text):,} characters")
    print(f"First 200 characters:\n{text[:200]!r}")
