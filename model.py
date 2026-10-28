"""
Mini LLM — full model, wiring embeddings + TransformerBlocks + output head.

Where this fits in the pipeline:

    BPE Token IDs
          |
    Token + Position Embeddings
          |
    TransformerBlocks (stack of TransformerBlock)   <-- transformer.py
          |
    Final LayerNorm
          |
    Vocabulary Projection (lm_head)
          |
        logits

This file does not reimplement attention or the transformer block — it
imports TransformerBlocks from transformer.py and wraps it with the
embedding layers on one side and the vocab head on the other, giving a
single end-to-end nn.Module you can call as model(token_ids).

Run this file directly to see shapes at every stage:

    python model.py
"""

import torch
import torch.nn as nn

import config
from transformer import TransformerBlocks


def get_device(name=None):
    """
    Resolve config.DEVICE (or `name`) to a torch.device. "auto" picks
    CUDA if available, then Apple-silicon MPS, then CPU.
    """
    name = name or config.DEVICE
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


class MiniLLM(nn.Module):
    """
    End-to-end GPT-style language model.

    forward(token_ids) -> logits over the vocabulary at every position,
    ready to be compared against target token ids with CrossEntropyLoss.
    """

    def __init__(self, vocab_size, embedding_dim=None, num_heads=None,
                 num_layers=None, max_seq_length=None, dropout=None,
                 tie_weights=None, gelu_approximate="none"):
        super().__init__()
        embedding_dim = embedding_dim or config.EMBEDDING_DIM
        max_seq_length = max_seq_length or config.MAX_SEQ_LENGTH
        dropout = config.DROPOUT if dropout is None else dropout
        tie_weights = config.TIE_WEIGHTS if tie_weights is None else tie_weights
        self.max_seq_length = max_seq_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.transformer = TransformerBlocks(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            dropout=dropout,
            gelu_approximate=gelu_approximate,
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        if tie_weights:
            # Weight tying: the output layer reuses the token-embedding
            # matrix. Both are (vocab_size, embedding_dim): the embedding
            # maps token id -> vector, lm_head scores a vector against
            # every token's row. Sharing one matrix means a token is
            # predicted when the hidden state looks like its embedding,
            # and saves vocab_size * embedding_dim parameters.
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    @classmethod
    def from_pretrained_gpt2(cls, name="gpt2", dropout=0.0):
        """
        Build a MiniLLM with OpenAI's pretrained GPT-2 weights.

        name: a Hugging Face GPT-2 model id -- "gpt2" (124M), "gpt2-medium"
              (355M), "gpt2-large" (774M) or "gpt2-xl" (1.5B). The sizes
              are read from its config, so all four work unchanged.

        Returns (model, model_config), model_config being the keyword
        arguments that rebuild the same architecture (what
        save_checkpoint() stores).

        MiniLLM was built the GPT-2 way, so this is a renaming plus one
        wrinkle: Hugging Face's GPT-2 uses Conv1D layers, which store
        their weight as (in_features, out_features) -- the transpose of
        nn.Linear's (out_features, in_features).

            Hugging Face key       MiniLLM key                 transpose
            wte / wpe              token_ / position_embedding
            h.{i}.ln_1 / ln_2      transformer.blocks.{i}.ln1 / ln2
            h.{i}.attn.c_attn      ...attn.qkv                 yes
            h.{i}.attn.c_proj      ...attn.proj                yes
            h.{i}.mlp.c_fc         ...ffn.net.0                yes
            h.{i}.mlp.c_proj       ...ffn.net.2                yes
            ln_f                   final_layer_norm
            lm_head                (tied to wte, nothing to copy)

        Needs the `transformers` package, only for downloading the weights.
        """
        from transformers import GPT2LMHeadModel

        hf_model = GPT2LMHeadModel.from_pretrained(name)
        hf_config = hf_model.config
        model_config = {
            "vocab_size": hf_config.vocab_size,
            "embedding_dim": hf_config.n_embd,
            "num_heads": hf_config.n_head,
            "num_layers": hf_config.n_layer,
            "max_seq_length": hf_config.n_positions,
            "dropout": dropout,
            "tie_weights": True,
            "gelu_approximate": "tanh",
        }
        model = cls(**model_config)

        rename = [
            ("transformer.wte.", "token_embedding."),
            ("transformer.wpe.", "position_embedding."),
            ("transformer.ln_f.", "final_layer_norm."),
            ("transformer.h.", "transformer.blocks."),
            (".ln_1.", ".ln1."), (".ln_2.", ".ln2."),
            (".attn.c_attn.", ".attn.qkv."), (".attn.c_proj.", ".attn.proj."),
            (".mlp.c_fc.", ".ffn.net.0."), (".mlp.c_proj.", ".ffn.net.2."),
        ]
        transposed = (".attn.qkv.weight", ".attn.proj.weight",
                      ".ffn.net.0.weight", ".ffn.net.2.weight")

        ours = model.state_dict()
        missing = set(ours) - {"lm_head.weight"}
        for key, tensor in hf_model.state_dict().items():
            # Older transformers versions also save the causal mask as
            # attn.bias / attn.masked_bias buffers; the fused attention
            # needs no mask, and lm_head is the tied wte.
            if key.endswith((".attn.bias", ".attn.masked_bias")) or key == "lm_head.weight":
                continue
            for old, new in rename:
                key = key.replace(old, new)
            if key.endswith(transposed):
                tensor = tensor.t()
            if ours[key].shape != tensor.shape:
                raise ValueError(f"{key}: expected {tuple(ours[key].shape)}, "
                                 f"got {tuple(tensor.shape)}")
            with torch.no_grad():
                ours[key].copy_(tensor)
            missing.discard(key)

        if missing:
            raise ValueError(f"No GPT-2 weights for: {sorted(missing)}")
        return model, model_config

    def _init_weights(self):
        """
        GPT-2 initialization.

        Every Linear and Embedding weight starts as N(0, 0.02), biases at 0
        (LayerNorm keeps its default gain 1, bias 0). PyTorch's own defaults
        are different for each layer type -- nn.Embedding even starts at
        std 1.0, which as tied output weights would give huge initial logits.

        The layers that write back into the residual stream (attention's
        output projection and the FFN's second Linear) are scaled down by
        1/sqrt(2 * num_layers). Each block adds two such outputs onto the
        residual, so without this the residual's variance grows with depth;
        with it, the sum over all 2 * num_layers additions stays about the
        same size no matter how many blocks are stacked.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        residual_std = 0.02 / (2 * len(self.transformer.blocks)) ** 0.5
        for block in self.transformer.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.net[2].weight, mean=0.0, std=residual_std)

    def optimizer_param_groups(self, weight_decay):
        """
        Split parameters for AdamW: weight matrices (2-D: Linear weights,
        embeddings, and the tied lm_head) get weight decay; LayerNorm
        gains and biases (1-D) don't. named_parameters() lists the tied
        embedding / lm_head matrix only once.
        """
        decay, no_decay = [], []
        for _, param in self.named_parameters():
            if param.requires_grad:
                (decay if param.dim() >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) of integer token ids
        Returns: logits, shape (batch, seq_len, vocab_size)
        """
        B, T = token_ids.shape

        position_ids = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        x = self.embed_dropout(x)

        x = self.transformer(x)

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None,
                 stop_ids=None):
        """
        Autoregressive sampling: predict one token, append it, repeat.

        token_ids:      (batch, seq_len) prompt token ids
        max_new_tokens: how many tokens to append, at most
        temperature:    divides the logits; 0 means greedy (argmax)
        top_k:          if set, sample only from the k most likely tokens
        stop_ids:       if set, stop early once every row's newest token is
                        one of these ids (e.g. a newline ending a chat
                        turn); that token is kept in the output

        Returns (batch, seq_len + new tokens), where new tokens is
        max_new_tokens unless stop_ids ended generation sooner. Call
        model.eval() first so dropout is off.
        """
        for _ in range(max_new_tokens):
            # The position embedding only has max_seq_length rows, so the
            # model can only ever look at the most recent window of tokens.
            context = token_ids[:, -self.max_seq_length:]

            # Only the last position's logits predict the *next* token.
            logits = self(context)[:, -1, :]  # (B, vocab_size)

            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth_best, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            token_ids = torch.cat([token_ids, next_id], dim=1)

            if stop_ids is not None and all(i in stop_ids for i in next_id[:, 0].tolist()):
                break

        return token_ids


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Any size works for a shape check; 4,069 is what the BPE tokenizer
    # learns on Tiny Shakespeare (4 special tokens + 65 base characters +
    # 4,000 merges).
    vocab_size = 4069
    model = MiniLLM(vocab_size=vocab_size)

    token_ids = torch.tensor([[1, 8]])  # (1, 2)
    print("Input shape:", token_ids.shape)

    logits = model(token_ids)
    print("Output shape:", logits.shape)
    assert logits.shape == (token_ids.shape[0], token_ids.shape[1], vocab_size)

    # also check the (1, 8) case explicitly, matching the plan
    token_ids_8 = torch.randint(0, vocab_size, (1, 8))
    logits_8 = model(token_ids_8)
    print("Input shape:", token_ids_8.shape, "-> Output shape:", logits_8.shape)
    assert logits_8.shape == (1, 8, vocab_size)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    print("\nOK — forward pass connected end to end.")
