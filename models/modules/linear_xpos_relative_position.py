# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, einsum


def fixed_pos_embedding(theta, positions):
    sinusoid_inp = einsum(positions, theta, 'n i, d -> n i d')
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin = repeat(scale * sin, '... d -> ... (d j)', j=2)
    cos = repeat(scale * cos, '... d -> ... (d j)', j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(self, heads, head_dim, layers):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.layers = layers
        d = self.head_dim // 2
        theta = torch.tensor(1.0 / (10000 ** (torch.arange(0, d) / d))).float()
        theta = repeat(theta, 'd -> h d', h=heads)
        self.theta = nn.Parameter(theta)

    def forward(self, x, positions, layer_num, downscale=False):
        N, S, D = x.size()
        assert positions.size() == (N, S)
        N0 = N // self.heads

        head_idx = torch.arange(1, self.heads + 1).float().to(x.device)
        h_factor = head_idx / self.heads
        l_factor = min(1, layer_num / self.layers)
        scale = torch.exp(- 8 * h_factor * (1 - l_factor))
        scale = repeat(scale, 'h -> (n h) o1 o2', n=N0, o1=1, o2=1)
        scale = torch.pow(scale, positions.unsqueeze(-1))

        assert scale.size() == (N, S, 1)

        theta = repeat(self.theta, 'h d -> (n h) d', n=N0)
        assert theta.size() == (N, D // 2)

        sinusoidal_inp = theta.unsqueeze(1) * positions.unsqueeze(-1)
        assert sinusoidal_inp.size() == (N, S, D // 2)
        sin, cos = torch.sin(sinusoidal_inp), torch.cos(sinusoidal_inp)

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        assert x.size() == (N, S, D)

        return x
