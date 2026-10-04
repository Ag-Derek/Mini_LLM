"""
Model tests: embeddings, causal attention, the full MiniLLM forward pass,
generation, and the single-batch overfit check that proves gradients
flow through the whole network.

Every model here is tiny (dims of 16-32) so the suite runs in seconds.
"""

import torch
import torch.nn as nn

from attention import CausalSelfAttentionHead, MultiHeadAttention
from embeddings import MyEmbedding
from model import MiniLLM, get_device

VOCAB = 20
SEQ = 8


def tiny_model(dropout=0.0):
    torch.manual_seed(0)
    return MiniLLM(vocab_size=VOCAB, embedding_dim=32, num_heads=4,
                   num_layers=2, max_seq_length=SEQ, dropout=dropout)


# ---------------------------------------------------------------------------
# Embeddings and attention
# ---------------------------------------------------------------------------

def test_my_embedding_matches_nn_embedding():
    mine = MyEmbedding(VOCAB, 16)
    ref = nn.Embedding(VOCAB, 16)
    ref.weight.data.copy_(mine.weight.data)

    ids = torch.randint(0, VOCAB, (3, 5))
    assert torch.equal(mine(ids), ref(ids))


def test_causal_head_never_attends_to_the_future():
    head = CausalSelfAttentionHead(embedding_dim=16, head_dim=8, max_seq_length=SEQ)
    _, weights = head(torch.randn(2, SEQ, 16))  # (B, T, T)

    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[:, future] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, SEQ))


def test_multi_head_attention_preserves_shape():
    mha = MultiHeadAttention(embedding_dim=16, num_heads=4, max_seq_length=SEQ)
    out, maps = mha(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
    assert len(maps) == 4


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def test_forward_shape():
    model = tiny_model()
    logits = model(torch.randint(0, VOCAB, (3, SEQ)))
    assert logits.shape == (3, SEQ, VOCAB)


def test_changing_future_tokens_does_not_change_past_logits():
    # The end-to-end version of the causal mask test: if any layer leaked
    # information backwards, editing the last token would change the
    # predictions at earlier positions.
    model = tiny_model().eval()
    ids = torch.randint(0, VOCAB, (1, SEQ))
    edited = ids.clone()
    edited[0, -1] = (ids[0, -1] + 1) % VOCAB

    with torch.no_grad():
        a, b = model(ids), model(edited)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_generate_runs_past_max_seq_length():
    model = tiny_model().eval()
    prompt = torch.randint(0, VOCAB, (2, 3))
    out = model.generate(prompt, max_new_tokens=3 * SEQ, temperature=1.0, top_k=5)

    assert out.shape == (2, 3 + 3 * SEQ)
    assert torch.equal(out[:, :3], prompt)
    assert out.min() >= 0 and out.max() < VOCAB


def test_greedy_generation_is_deterministic_and_matches_top_k_1():
    model = tiny_model().eval()
    prompt = torch.randint(0, VOCAB, (1, 4))

    greedy = model.generate(prompt, 10, temperature=0)
    assert torch.equal(greedy, model.generate(prompt, 10, temperature=0))
    # Sampling from only the single best token is greedy decoding.
    assert torch.equal(greedy, model.generate(prompt, 10, temperature=1.0, top_k=1))


def test_single_batch_overfit():
    # If gradients reach every layer, a model can memorize one fixed batch:
    # loss should fall from ~ln(VOCAB) ~ 3.0 to near zero.
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    X = torch.randint(0, VOCAB, (4, SEQ))
    Y = torch.randint(0, VOCAB, (4, SEQ))

    for _ in range(150):
        logits = model(X)
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB), Y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1


def test_get_device_explicit_cpu():
    assert get_device("cpu") == torch.device("cpu")
    assert get_device("auto").type in {"cpu", "cuda", "mps"}
