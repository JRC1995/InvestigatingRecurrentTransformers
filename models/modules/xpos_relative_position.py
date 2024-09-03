# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, einsum


def fixed_pos_embedding(d, positions):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d) / d))
    sinusoid_inp = einsum(positions, inv_freq.float().to(positions.device), 'n i, d -> n i d')
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
    def __init__(
            self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, positions, downscale=False):
        length = x.shape[1]
        min_pos = -length // 2
        N, _ = positions.size()
        d = x.size(-1) // 2

        scale = torch.pow(self.scale.view(1, 1, d), (positions.unsqueeze(-1) + min_pos) / self.scale_base)
        assert scale.size() == (N, length, d)
        sin, cos = fixed_pos_embedding(d, positions)

        assert sin.size() == (N, length, d)
        assert cos.size() == (N, length, d)

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
