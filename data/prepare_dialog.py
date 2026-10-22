"""
data/prepare_dialog.py
======================

Download the DailyDialog dataset and write it as a plain-text chat corpus
that train.py can learn from.

Why this exists:
    The model only learns the layout of the text it is trained on. Tiny
    Shakespeare is plays, so the model writes plays. To get replies
    instead, it has to be trained on conversations laid out the same way
    chat mode will prompt it (see config.CHAT_USER_PREFIX):

        User: Say, Jim, how about going for a few beers after dinner?
        Bot: You know that is tempting but is really not good for our fitness.
        User: What do you mean? It will help us to relax.

        User: <next conversation>
        ...

Source:
    DailyDialog (Li et al., 2017, https://aclanthology.org/I17-1099/):
    ~13K everyday English conversations. The original download link
    (yanran.li) no longer serves the files, so this uses the unchanged
    copy at huggingface.co/datasets/roskoN/dailydialog. Licensed CC
    BY-NC-SA 4.0: fine for learning and research, not for commercial use.

The raw files have one conversation per line, turns separated by
"__eou__", and spaces around all punctuation ("I ’ m here , ok ?").
clean_utterance() undoes that so the tokenizer sees normal English.

Run from the project root (like data/corpus.py, as a module):

    python -m data.prepare_dialog
"""

import argparse
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import config

DOWNLOAD_URL = "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/{split}.zip"

# Written in this order so the contiguous 90/10 split in train.py puts
# (mostly) the dataset's own validation and test conversations last.
SPLITS = ("train", "validation", "test")

TURN_SEPARATOR = "__eou__"

# Non-ASCII characters in the raw text and their plain replacements. All
# but a handful of occurrences are curly apostrophes; the rest would
# otherwise each take a vocab slot the model sees too rarely to learn.
CHAR_REPLACEMENTS = {
    "’": "'", "‘": "'", "′": "'",   # ’ ‘ ′
    "“": '"', "”": '"',                  # “ ”
    "—": "-", "–": "-",                  # — –
    "。": ".", "、": ",",                  # Chinese full stop, comma
}


def clean_utterance(text: str) -> str:
    """
    Turn one raw DailyDialog turn into normal English text:

        "I ’ m here , ok ?"  ->  "I'm here, ok?"
    """
    for old, new in CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = " ".join(text.split())  # collapse runs of whitespace

    # Rejoin contractions split around the apostrophe: "don ' t" -> "don't".
    # The raw text sometimes capitalizes the suffix ("I ' Ve"), so lower it.
    text = re.sub(r"(\w) ' (\w+)", lambda m: f"{m[1]}'{m[2].lower()}", text)
    # No space before closing punctuation, none inside brackets, and
    # "$ 5" -> "$5".
    text = re.sub(r" ([.,?!;:%)])", r"\1", text)
    text = re.sub(r"\( ", "(", text)
    text = re.sub(r"\$ (\d)", r"$\1", text)
    return text


def format_dialogue(utterances: list[str]) -> str:
    """
    Lay a conversation out as alternating User / Bot lines. Who is "User"
    is arbitrary in DailyDialog (two people talking), so the first speaker
    is always User; this gives the model the most examples of a Bot line
    following a User line, which is what chat mode asks it to write.
    """
    prefixes = (config.CHAT_USER_PREFIX, config.CHAT_BOT_PREFIX)
    return "\n".join(f"{prefixes[i % 2]} {u}" for i, u in enumerate(utterances))


def parse_raw_dialogues(raw_text: str) -> list[list[str]]:
    """Split a raw dialogues_<split>.txt file into cleaned turns per conversation."""
    dialogues = []
    for line in raw_text.splitlines():
        turns = [clean_utterance(t) for t in line.split(TURN_SEPARATOR)]
        turns = [t for t in turns if t]
        if len(turns) >= 2:  # a single turn isn't a conversation
            dialogues.append(turns)
    return dialogues


def build_corpus(dialogues: list[list[str]]) -> tuple[str, int]:
    """
    Join conversations into one corpus string, separated by blank lines.
    DailyDialog repeats some conversations (also across its splits), so
    exact duplicates are dropped. Returns (corpus, duplicates_removed).
    """
    seen, blocks = set(), []
    for turns in dialogues:
        block = format_dialogue(turns)
        if block not in seen:
            seen.add(block)
            blocks.append(block)
    return "\n\n".join(blocks) + "\n", len(dialogues) - len(blocks)


def download_split(split: str) -> str:
    """Download one split's zip and return its dialogues_<split>.txt text."""
    url = DOWNLOAD_URL.format(split=split)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    return archive.read(f"{split}/dialogues_{split}.txt").decode("utf-8")


def default_output_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / config.DATA_DIR / config.DIALOG_CORPUS_FILENAME


def parse_args():
    parser = argparse.ArgumentParser(description="Build the DailyDialog chat corpus.")
    parser.add_argument("--output", default=None,
                        help="where to write the corpus "
                             "(default: config.DATA_DIR / config.DIALOG_CORPUS_FILENAME)")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output) if args.output else default_output_path()

    dialogues = []
    for split in SPLITS:
        dialogues.extend(parse_raw_dialogues(download_split(split)))

    corpus, duplicates = build_corpus(dialogues)
    output.write_text(corpus, encoding="utf-8")

    # Windows consoles often can't print every character; never crash on it.
    sys.stdout.reconfigure(errors="replace")
    num_lines = corpus.count("\n")
    print(f"\nWrote {output}")
    print(f"  conversations: {len(dialogues) - duplicates:,} "
          f"({duplicates:,} duplicates removed)")
    print(f"  size:          {len(corpus):,} characters, {num_lines:,} lines "
          f"(Tiny Shakespeare: 1,115,394 characters)")
    print(f"\nFirst conversations:\n\n{corpus[:600]}")


if __name__ == "__main__":
    main()
