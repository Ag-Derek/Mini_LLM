"""
dataset.py
----------
Turns raw text into the (input, target) training pairs a Transformer needs,
using a sliding window over token IDs.

Pipeline recap:

    Raw Text -> Tokenizer -> Token IDs -> Dataset -> Training Samples

This file assumes you already have one of yesterday's tokenizers
(tokenizer.py / word_tokenizer.py / bpe_tokenizer.py), each of which is
expected to expose an `encode(text) -> list[int]` method. Any object with
that interface will work here — the dataset doesn't care which tokenizer
produced the IDs.
"""

from torch.utils.data import Dataset, DataLoader
import torch


class TextDataset(Dataset):
    """
    Wraps a token-ID sequence and slices it into fixed-length,
    overlapping (input, target) windows for next-token prediction.

    Example (context_length=4, stride=1):

        tokens = [5, 18, 9, 42, 9, 31, 70]

        window 1: input=[5,18,9,42]   target=[18,9,42,9]
        window 2: input=[18,9,42,9]   target=[9,42,9,31]
        window 3: input=[9,42,9,31]   target=[42,9,31,70]
    """

    def __init__(self, text, tokenizer, context_length=8, stride=1):
        """
        text:           raw string to train on
        tokenizer:      any object exposing .encode(text) -> list[int]
        context_length: how many tokens the model sees at once
        stride:         how far the window moves each step
                        (stride == context_length gives non-overlapping
                        chunks; stride == 1 gives maximum overlap / data reuse)
        """
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride

        token_ids = tokenizer.encode(text)

        if len(token_ids) < context_length + 1:
            raise ValueError(
                f"Text produced only {len(token_ids)} tokens, but at least "
                f"{context_length + 1} are needed for one training sample "
                f"(context_length={context_length}). Use a longer text or a "
                f"smaller context_length."
            )

        self.input_ids = []
        self.target_ids = []

        # Slide a window of size `context_length` across the token stream.
        # Each step, input = window, target = window shifted one token right.
        for start in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[start : start + context_length]
            target_chunk = token_ids[start + 1 : start + context_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        """How many samples exist."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Give me sample number `idx`."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    tokenizer,
    context_length=8,
    stride=1,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Convenience wrapper: builds a TextDataset and hands back a DataLoader
    ready to feed a Transformer.

        batch = next(iter(dataloader))
        inputs, targets = batch   # each shaped [batch_size, context_length]
    """
    dataset = TextDataset(text, tokenizer, context_length=context_length, stride=stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def load_text(path):
    """Load raw text from a file (utf-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Minimal smoke test using a tiny stand-in tokenizer, so this file
    # runs standalone even before tokenizer.py is wired in. Swap in your
    # real tokenizer (CharTokenizer / WordTokenizer / BPETokenizer) once
    # it's available.
    # ------------------------------------------------------------------
    class DummyCharTokenizer:
        """Encodes each character as its ordinal value. Just for testing."""

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    sample_text = "I love AI because AI is amazing. " * 20

    from bpe_tokenizer import BPETokenizer

    tokenizer = BPETokenizer()
    tokenizer.train(sample_text, num_merges=100)
    print(f"BPE vocab size: {tokenizer.vocab_size}")
    context_length = 8
    stride = 4

    dataset = TextDataset(sample_text, tokenizer, context_length=context_length, stride=stride)
    print(f"Number of samples: {len(dataset)}")

    first_input, first_target = dataset[0]
    print(f"Sample 0 input:  {first_input.tolist()}")
    print(f"Sample 0 target: {first_target.tolist()}")

    dataloader = create_dataloader(
        sample_text, tokenizer, context_length=context_length, stride=stride, batch_size=4
    )
    batch_inputs, batch_targets = next(iter(dataloader))
    print(f"Batch inputs shape:  {batch_inputs.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")
