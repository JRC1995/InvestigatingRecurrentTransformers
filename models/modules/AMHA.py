import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, einsum
from .xpos_relative_position import XPOS


class AMHA(nn.Module):
    def __init__(self, config):
        super(AMHA, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.heads = config["heads"]
        self.head_dim = config["head_dim"]
        self.dropout = nn.Dropout(config["attn_dropout"])
        self.scaling = self.head_dim ** -0.5
        self.config = config

        self.k_proj = nn.Linear(self.hidden_size, self.heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.heads * self.head_dim, bias=True)
        self.q_proj = nn.Linear(self.hidden_size, self.heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.heads * self.head_dim, self.hidden_size, bias=True)
        self.xpos = XPOS(self.head_dim)

        # inv_freq = 1.0 / (10000 ** (T.arange(0, self.head_dim) / self.head_dim))
        # self.theta = nn.Parameter(T.tensor(inv_freq))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)

    """
    Forward Function
    """

    def masked_softmax(self, logits, mask, dim=-1):
        if mask is None:
            return F.softmax(logits, dim=dim)

        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        logits = F.softmax(logits, dim=dim)
        return logits

    # %%
    def forward(self, sequence, positions, input_mask, layer_num,
                kv_sequence=None, kv_positions=None,
                query_input_mask=None,
                position_encodings=None,
                kv_position_encodings=None,
                xpos=True):
        """
        :param sequence: sequence: batch x seq_len x dimension
        :param positions: batch_size x seq_len
        :param position_encodings: batch_size x seq_len x dimensions
        :param input_mask: input_mask:batch x seq_len
        :param layer_num: scalar (current layer number)
        :return:{attended_values: batch_size x seq_lean x dimension}
        """

        if position_encodings is None:
            position_encodings = 0

        if kv_sequence is None:
            kv_sequence = sequence
            kv_positions = positions
            kv_position_encodings = position_encodings
            query_input_mask = input_mask

        if position_encodings is None:
            position_encodings = 0
        if kv_position_encodings is None:
            kv_position_encodings = 0

        N, S, D = sequence.size()
        _, S2, _ = kv_sequence.size()
        assert input_mask.size() == (N, S2)

        assert S == S2
        attention_mask = repeat(input_mask, 'n s -> n c s', c=S)
        assert attention_mask.size() == (N, S, S2)
        attention_mask = T.tril(attention_mask, diagonal=0)
        attention_mask = attention_mask.unsqueeze(1)

        Q = rearrange(self.q_proj(sequence + position_encodings), 'n s (h d) -> (n h) s d', h=self.heads)
        K = rearrange(self.k_proj(sequence + kv_position_encodings), 'n s (h d) -> (n h) s d', h=self.heads)
        V = rearrange(self.v_proj(sequence), 'n s (h d) -> (n h) s d', h=self.heads)
        assert Q.size() == (N * self.heads, S, self.head_dim)
        Q *= self.scaling

        positions = repeat(positions, 'n s -> (n h) s', h=self.heads)
        kv_positions = repeat(kv_positions, 'n s -> (n h) s', h=self.heads)

        if xpos:
            K = self.xpos(K, kv_positions, downscale=True)
            Q = self.xpos(Q, positions, downscale=False)

        assert Q.size() == (N * self.heads, S, self.head_dim)
        assert K.size() == (N * self.heads, S2, self.head_dim)
        assert V.size() == (N * self.heads, S2, self.head_dim)

        A = einsum(Q, K, 'n s1 d, n s2 d -> n s1 s2')
        assert A.size() == (N * self.heads, S, S2)

        # A = A * D
        A = A.view(N, self.heads, S, S2)
        V = V.view(N, self.heads, S2, self.head_dim)

        A = T.sigmoid(A) * attention_mask
        assert A.size() == (N, self.heads, S, S2)

        C = T.exp(T.cumsum(T.log(1 - T.flip(A, dims=[2, 3]) + 1e-8), dim=-1))
        C = T.flip(C, dims=[2, 3])
        C = T.cat([T.ones(N, self.heads, S, 1).float().to(C.device),
                   C[..., :-1]], dim=-1)
        assert C.size() == (N, self.heads, S, S2)
        A = self.dropout(C * A)

        attended_values = T.matmul(A, V)
        # print(attended_values.size())
        assert attended_values.size() == (N, self.heads, S, self.head_dim)
        attended_values = rearrange(attended_values, 'n h s d -> n s (h d)', h=self.heads)
        assert attended_values.size() == (N, S, self.heads * self.head_dim)

        out = self.out_proj(attended_values)
        assert out.size() == (N, S, D)

        return {"attended_values": out}
