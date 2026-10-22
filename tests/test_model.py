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


def test_weight_tying_shares_one_matrix():
    torch.manual_seed(0)
    kwargs = dict(vocab_size=VOCAB, embedding_dim=32, num_heads=4,
                  num_layers=2, max_seq_length=SEQ)
    tied = MiniLLM(**kwargs, tie_weights=True)
    untied = MiniLLM(**kwargs, tie_weights=False)

    assert tied.lm_head.weight is tied.token_embedding.weight
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(untied) - count(tied) == VOCAB * 32

    # One gradient step on the output side changes the embeddings too.
    before = tied.token_embedding.weight.detach().clone()
    tied.lm_head(torch.randn(3, 32)).sum().backward()
    torch.optim.SGD(tied.parameters(), lr=0.1).step()
    assert not torch.equal(tied.token_embedding.weight, before)


def test_gpt2_initialization():
    torch.manual_seed(0)
    model = MiniLLM(vocab_size=500, embedding_dim=128, num_heads=4,
                    num_layers=8, max_seq_length=SEQ)
    block = model.transformer.blocks[0]

    # Weights start at std 0.02 (embeddings included, not nn.Embedding's 1.0)...
    for weight in (model.token_embedding.weight, model.position_embedding.weight,
                   block.attn.heads[0].query.weight, block.ffn.net[0].weight):
        assert abs(weight.std().item() - 0.02) < 0.003
    assert torch.all(block.ffn.net[0].bias == 0)

    # ...except the projections back into the residual stream, which are
    # scaled down by 1/sqrt(2 * num_layers) = 1/4 here.
    for weight in (block.attn.proj.weight, block.ffn.net[2].weight):
        assert abs(weight.std().item() - 0.005) < 0.001

    # LayerNorm keeps its default gain 1, bias 0.
    assert torch.all(block.ln1.weight == 1) and torch.all(block.ln1.bias == 0)


def test_weight_decay_only_on_weight_matrices():
    model = tiny_model()
    decay, no_decay = model.optimizer_param_groups(weight_decay=0.1)

    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert any(p is model.transformer.blocks[0].ln1.weight for p in no_decay["params"])

    # Every parameter lands in exactly one group, the tied matrix only once.
    grouped = decay["params"] + no_decay["params"]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}


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


def test_generate_stops_at_a_stop_token():
    model = tiny_model().eval()
    prompt = torch.randint(0, VOCAB, (1, 4))
    full = model.generate(prompt, 10, temperature=0)

    # Make the third generated token a stop token: generation ends right
    # after it (keeping it), and nothing before it changes.
    stop_id = full[0, 4 + 2].item()
    first = (full[0, 4:] == stop_id).nonzero()[0].item()  # it may appear earlier
    stopped = model.generate(prompt, 10, temperature=0, stop_ids={stop_id})
    assert torch.equal(stopped, full[:, :4 + first + 1])


def test_generate_without_stop_tokens_runs_to_max_new_tokens():
    model = tiny_model().eval()
    prompt = torch.randint(0, VOCAB, (1, 4))
    out = model.generate(prompt, 10, temperature=0, stop_ids=set())
    assert out.shape == (1, 14)


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
