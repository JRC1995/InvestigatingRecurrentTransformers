import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from models.modules import *


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.layers = config["layers"]
        self.hidden_size = config["hidden_size"]

        self.START = nn.Parameter(T.randn(self.hidden_size))
        self.END = nn.Parameter(T.randn(self.hidden_size))
        self.leaf_transform = config["leaf_transform"]

        if self.leaf_transform:
            self.leaf_transform_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                      nn.LayerNorm(self.hidden_size))

        self.encoderBlock = eval(config["block"])
        self.TransformerLayers = nn.ModuleList([self.encoderBlock(config) for _ in range(self.layers)])

        if self.config["pooling"] == "cls+":
            self.cls_compress = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.eps = 1e-8

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%
    def augment_sequence(self, sequence, input_mask):
        N, S, D = sequence.size()
        assert input_mask.size() == (N, S)

        """
        AUGMENT SEQUENCE WITH START AND END TOKENS
        """
        # ADD START TOKEN
        START = repeat(self.START, 'd -> n s d', n=N, s=1)
        sequence = T.cat([START, sequence], dim=1)
        assert sequence.size() == (N, S + 1, D)
        input_mask = T.cat([T.ones(N, 1).to(input_mask.dtype).to(input_mask.device),
                            input_mask], dim=1)
        assert input_mask.size() == (N, S + 1)

        # ADD END TOKEN
        input_mask_no_end = T.cat([input_mask.clone(),
                                   T.zeros(N, 1).to(input_mask.dtype).to(input_mask.device)], dim=1)
        input_mask_yes_end = T.cat([T.ones(N, 1).to(input_mask.dtype).to(input_mask.device),
                                    input_mask.clone()], dim=1)
        END_mask = input_mask_yes_end - input_mask_no_end
        assert END_mask.size() == (N, S + 2)
        END_mask = END_mask.unsqueeze(-1)

        END = repeat(self.END, 'd -> n s d', n=N, s=1)
        sequence = T.cat([sequence, T.zeros(N, 1, D).to(sequence.dtype).to(sequence.device)], dim=1)
        sequence = END_mask * END + (1 - END_mask) * sequence
        input_mask = input_mask_yes_end

        return sequence, input_mask, END_mask

    # %%
    def forward(self, sequence,  input_mask):
        """
        :param sequence: batch_size x seq_len x dimensions
        :param input_mask: batch_size x seq_len
        :return: {sequence: batch_size x seq_len x dimensions, 
                  global_state: batch_size x dimensions, 
                  input_mask: batch_size x seq_len,
                  aux_len: scalar or None}
        """
        sequence, input_mask, END_mask = self.augment_sequence(sequence, input_mask)
        N, S, D = sequence.size()
        positions = T.cumsum(input_mask, dim=1) - 1

        if self.leaf_transform:
            sequence = self.leaf_transform_layer(sequence)

        for l in range(self.layers):
            sequence = self.TransformerLayers[l](sequence=sequence,
                                                 kv_sequence=sequence,
                                                 positions=positions,
                                                 kv_positions=positions,
                                                 input_mask=input_mask,
                                                 query_input_mask=input_mask,
                                                 layer_num=l+1)["sequence"]
            # layer_num l + oned to index it from 1.
            assert sequence.size() == (N, S, D)

        if self.config["pooling"] == "cls":
            global_state = sequence[:, 0, :]
        elif self.config["pooling"] == "cls+":
            start = sequence[:, 0, :]
            end = T.sum(sequence * END_mask, dim=1)
            global_state = self.cls_compress(T.cat([start, end], dim=-1))
        elif self.config["pooling"] == "mean":
            global_state = T.sum(sequence * input_mask.unsqueeze(-1), dim=1) / T.sum(input_mask.unsqueeze(-1), dim=1)

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": None}