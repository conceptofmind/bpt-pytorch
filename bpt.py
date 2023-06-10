import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# feedforward network


class FFN(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        dropout, 
        act=nn.GELU
    ):
        super().__init__()
        self.fc_in = nn.Linear(dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, dim)
        self.act = act()
        self.resid_dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(dim)

    def forward(self, hidden_states):
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.resid_dropout(hidden_states)
        return hidden_states

def blockwise_compute_ffn(cell, inputs, chunk_size):
    inputs = rearrange(inputs, 'b (n c) d -> b n c d', c=chunk_size)
    inputs = rearrange(inputs, 'b n c d -> n b c d')
    num_q, _, _, _ = inputs.shape
    res = []
    for i in range(num_q):
        res.append(cell(inputs[i]))
    res = torch.stack(res, dim=0)
    res = rearrange(res, 'n b c d -> b (n c) d')
    return res

# Attention


class Attention(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        intermediate_size, 
        resid_pdrop = 0.0,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.head_dim = self.embed_dim // num_heads

        self.to_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.layer_norm_1 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(resid_pdrop)

        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        b, h, n, d, device = *x.shape, x.device

        x = self.layer_norm_1(x)

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # split heads
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (query, key, value))

        # rotary embeddings
        positions = self.get_rotary_embedding(n, device)
        query, key = map(lambda t: apply_rotary_pos_emb(positions, t), (query, key))

        q_len = query.shape[1]
        kv_len = key.shape[1]



def blockwise_cross_entropy(logits, tokens, valid=None, chunk_size=None):
    if valid is None:
        valid = torch.ones(tokens.shape[:2], device=logits.device)
    valid = valid.float()
    logits = logits.view(-1, logits.shape[-1])
    tokens = tokens.view(-1)
    valid = valid.view(-1)

    def _cross_entropy_loss_and_accuracy(logits, tokens, valid):
        valid_text_length = torch.max(valid.sum(dim=-1), torch.tensor(1e-10).to(logits.device))

        token_log_prob = F.log_softmax(logits, dim=-1)
        token_log_prob = token_log_prob[torch.arange(len(tokens)), tokens]
        token_log_prob = torch.where(valid > 0.0, token_log_prob, torch.tensor(0.0).to(logits.device))
        correct = torch.where(
            valid > 0.0,
            torch.argmax(logits, dim=-1) == tokens,
            torch.tensor(False).to(logits.device)
        )
        return token_log_prob, correct.float(), valid_text_length

    num_chunk = logits.shape[0] // chunk_size
    logits = rearrange(logits, '(n c) d -> n c d', c=chunk_size)
    tokens = rearrange(tokens, '(n c) -> n c', c=chunk_size)
    valid = rearrange(valid, '(n c) -> n c', c=chunk_size)

    loss, accuracy, num = 0.0, 0.0, 0
    for i in range(num_chunk):
        token_log_prob, correct, valid_text_length = _cross_entropy_loss_and_accuracy(logits[i], tokens[i], valid[i])
        loss += token_log_prob.sum() / valid_text_length
        accuracy += correct.sum() / valid_text_length
        num = num + 1

    loss = - loss / num
    accuracy = accuracy / num
    return loss, accuracy


class Blockwise_LM_Head(nn.Module):
    def __init__(self, vocab_size, chunk_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.lm_head = nn.Linear(
            chunk_size, 
            vocab_size, 
            bias=True
        )

    def forward(self, inputs):
        inputs = rearrange(inputs, 'b (n c) d -> b n c d', c=self.chunk_size)
        inputs = rearrange(inputs, 'b n c d -> n b c d')
        num_q, _, _, _ = inputs.shape
        res = []
        for i in range(num_q):
            res.append(self.lm_head(inputs[i]))
        res = torch.stack(res, dim=0)
        res = rearrange(res, 'n b c d -> b (n c) d')
        return res