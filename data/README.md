# data/

## Corpus: Tiny Shakespeare

`tiny_corpus.txt` is the "Tiny Shakespeare" dataset -- a ~1.1 MB, 40,000-line
plain-text concatenation of Shakespeare's plays, originally assembled by
Andrej Karpathy for the char-rnn project and widely reused since as a
standard small benchmark for character- and subword-level language models.

- **Source**: `karpathy/char-rnn`, `data/tinyshakespeare/input.txt`
- **License**: the underlying plays are public domain; this particular
  compiled text file is the version popularized by the char-rnn repo and
  used in the same form by many tutorials (e.g. nanoGPT), which is why it's
  a good choice here -- results are easy to sanity-check against other
  implementations.
- **Size**: 1,115,394 characters / 40,000 lines.

Chosen over alternatives (Alice in Wonderland, Sherlock Holmes) because it's
the standard reference corpus for small educational Transformer builds, so
our tokenizer/vocab/loss numbers can be roughly compared against known
public results at the same scale.

## corpus.py

`load_corpus()` returns the raw corpus text as a single string -- no
lowercasing, splitting, or tokenizing. Those decisions belong to whichever
tokenizer/dataset script consumes the text, not to the loader.

```python
from data.corpus import load_corpus

text = load_corpus()
```

Path resolution: `corpus.py` finds `tiny_corpus.txt` relative to the project
root (not the current working directory), using `config.DATA_DIR` and
`config.CORPUS_FILENAME`. This means:

- Importing `from data.corpus import load_corpus` from anywhere in the
  project works correctly.
- Running `python -m data.corpus` from the project root works (this is the
  correct way to run a module that lives inside a package and imports a
  sibling top-level module like `config`).
- Running `python data/corpus.py` directly does **not** work -- Python adds
  `data/`, not the project root, to `sys.path` when a script is invoked
  that way, so `import config` fails. This is a standard Python packaging
  quirk, not a bug in the loader.
