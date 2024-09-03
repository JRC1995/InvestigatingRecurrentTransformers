import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from typing import Optional, Tuple, Union
import math


def generate_absolute_positional_embeddings(max_len, d_model, freeze=True):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, d_model)
        position = T.arange(1, max_len + 1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (max_len, d_model)
        pe = pe / math.sqrt(d_model)
    return pe, nn.Embedding.from_pretrained(pe,
                                            freeze=freeze)


def SRMSNorm(x, eps=1e-8):
    d = x.size(-1)
    scale = math.sqrt(d)
    x_norm = T.norm(x, dim=-1).unsqueeze(-1) / scale
    return x / (x_norm + eps)


def laplace_act(x):
    mu = math.sqrt(0.5)
    std = math.sqrt((4 * math.pi) ** -1)
    return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5


def sum_normalize(logits, dim=-1):
    eps = 1e-8
    return logits / T.sum(logits + eps, keepdim=True, dim=dim)


def masked_softmax(logits, mask=None, dim=-1):
    eps = 1e-20
    probs = F.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs


def reverse(state, count_zeros, count_zeros_end):
    with T.no_grad():
        N, S, D = state.size()
        reverse_state = T.flip(state, dims=[1])
        reverse_state = T.cat([reverse_state,
                               T.zeros(N, S, D).float().to(state.device)], dim=1)
        new_batch_stack = []
        for i in range(N):
            start_id = count_zeros[i]
            end_id = count_zeros_end[i]
            new_batch_stack.append(reverse_state[i, start_id:end_id, :])
        reverse_state = T.stack(new_batch_stack, dim=0)
        return reverse_state
