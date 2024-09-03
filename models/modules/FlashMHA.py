import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, einsum
from .xpos_relative_position import XPOS
from .bert_padding import pad_input, unpad_input
from flash_attn import flash_attn_varlen_kvpacked_func


class FlashMHA(nn.Module):
    def __init__(self, config):
        super(FlashMHA, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.heads = config["heads"]
        self.head_dim = config["head_dim"]
        self.dropout = nn.Dropout(config["attn_dropout"])
        self.scaling = self.head_dim ** -0.5
        self.config = config
        self.ex_halt = self.config["ex_halt"]

        if self.config["ex_halt"]:
            self.qhalt_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, (self.heads + 1) * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, (self.heads + 1) * self.head_dim, bias=False)
            self.outhalt_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        else:
            self.k_proj = nn.Linear(self.hidden_size, self.heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, self.hidden_size, bias=False)
        self.xpos = XPOS(self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        if self.ex_halt:
            nn.init.xavier_uniform_(self.qhalt_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.outhalt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # nn.init.constant_(self.out_proj.bias, 0.0)
        # nn.init.constant_(self.q_proj.bias, 0.0)
        # nn.init.constant_(self.k_proj.bias, 0.0)
        # nn.init.constant_(self.v_proj.bias, 0.0)

    """
    Forward Function
    """

    def flash(self, q, kv, query_padding_mask, key_padding_mask):
        # print("hello")
        q = q.to(T.float16)
        kv = kv.to(T.float16)

        qseqlen = q.size(1)
        kseqlen = kv.size(1)
        if qseqlen == kseqlen:
            causal = self.config["causal"]
        else:
            causal = False

        # adapted from here: https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L54
        batch_size, seqlen, _, nheads, _ = kv.size()
        kv = rearrange(kv, 'b s two h d -> b s (two h d)')
        kv_unpad, kv_indices, cu_seqlens_k, max_seqlen_k = unpad_input(kv, key_padding_mask.bool())
        kv_unpad = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)

        batch_size, seqlen, nheads, _ = q.size()
        q = rearrange(q, 'b s h d -> b s (h d)')
        q_unpad, q_indices, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask.bool())
        q_unpad = rearrange(q_unpad, 'nnz (h d) -> nnz h d', h=nheads)

        output_unpad = flash_attn_varlen_kvpacked_func(q=q_unpad,
                                                       kv=kv_unpad,
                                                       cu_seqlens_q=cu_seqlens_q,
                                                       cu_seqlens_k=cu_seqlens_k,
                                                       max_seqlen_q=max_seqlen_q,
                                                       max_seqlen_k=max_seqlen_k,
                                                       dropout_p=self.config["attn_dropout"] if self.training else 0.0,
                                                       softmax_scale=self.scaling,
                                                       causal=causal)

        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                     q_indices, batch_size, seqlen),
                           'b s (h d) -> b s h d', h=nheads)

        return output

    # %%
    def forward(self, sequence, positions, input_mask, layer_num,
                halt_sequence=None,
                halt_position_encodings=None,
                kv_sequence=None, kv_positions=None,
                query_input_mask=None,
                position_encodings=None,
                kv_position_encodings=None,
                xpos=True):
        """
        :param sequence: sequence: batch x seq_len x dimension
        :param positions: batch_size x seq_len
        :param position_encodings: batch_size x seq_len x dimensions
        :param attention_mask: attention_mask:batch x seq_len x seqlen
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
        if halt_position_encodings is None:
            halt_position_encodings = 0

        N, S, D = sequence.size()
        _, S2, _ = kv_sequence.size()

        assert input_mask.size() == (N, S2)

        if self.ex_halt:
            Q = rearrange(self.q_proj(sequence + position_encodings), 'n s (h d) -> n s h d', h=self.heads)
            Qhalt = rearrange(self.qhalt_proj(halt_sequence + halt_position_encodings),
                              'n s (h d) -> n s h d', h=1)
            Q = T.cat([Q, Qhalt], dim=-2)
            Q = rearrange(Q, 'n s h d -> (n h) s d', h=self.heads+1)
            self.heads = self.heads + 1
        else:
            Q = rearrange(self.q_proj(sequence + position_encodings), 'n s (h d) -> (n h) s d', h=self.heads)

        K = rearrange(self.k_proj(kv_sequence + kv_position_encodings), 'n s (h d) -> (n h) s d', h=self.heads)
        V = rearrange(self.v_proj(kv_sequence), 'n s (h d) -> (n h) s d', h=self.heads)
        assert Q.size() == (N * self.heads, S, self.head_dim)

        if positions is not None:
            positions = repeat(positions, 'n s -> (n h) s', h=self.heads)
        if kv_positions is not None:
            kv_positions = repeat(kv_positions, 'n s -> (n h) s', h=self.heads)

        if xpos:
            K = self.xpos(K, kv_positions, downscale=True)
            Q = self.xpos(Q, positions, downscale=False)

        assert Q.size() == (N * self.heads, S, self.head_dim)
        assert K.size() == (N * self.heads, S2, self.head_dim)
        assert V.size() == (N * self.heads, S2, self.head_dim)

        kv = T.stack([K, V], dim=0)
        assert kv.size() == (2, N * self.heads, S2, self.head_dim)
        kv = rearrange(kv, 'c (n h) s d -> n s c h d', h=self.heads, c=2)
        q = rearrange(Q, '(n h) s d -> n s h d', h=self.heads)

        attended_values = self.flash(q=q, kv=kv,
                                     query_padding_mask=query_input_mask, key_padding_mask=input_mask)
        assert attended_values.size() == (N, S, self.heads, self.head_dim)
        if self.ex_halt:
            halt_sequence = attended_values[:, :, -1, :]
            halt_sequence = self.outhalt_proj(halt_sequence)
            attended_values = attended_values[:, :, 0:-1, :]
            self.heads = self.heads - 1

        attended_values = rearrange(attended_values, 'n s h d -> n s (h d)', h=self.heads)

        """
        Q = Q.view(N, self.heads, S, self.head_dim)
        K = K.view(N, self.heads, S, self.head_dim)
        V = V.view(N, self.heads, S, self.head_dim)

        with T.backends.cuda.sdp_kernel(enable_math=False):
            attended_values = F.scaled_dot_product_attention(Q, K, V, attn_mask=repeat(input_mask, 'n s -> n h c s',
                                                                                       h=self.heads, c=S))
            assert attended_values.size() == (N, self.heads, S, self.head_dim)
            attended_values = rearrange(attended_values, 'n h s d -> n s (h d)')
        """

        assert attended_values.size() == (N, S, self.heads * self.head_dim)

        out = self.out_proj(attended_values)
        assert out.size() == (N, S, D)

        return {"attended_values": out, "halt_sequence": halt_sequence}
