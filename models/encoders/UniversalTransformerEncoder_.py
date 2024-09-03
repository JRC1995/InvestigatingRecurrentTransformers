import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from models.modules import *


class UniversalTransformerEncoder(nn.Module):
    def __init__(self, config):
        super(UniversalTransformerEncoder, self).__init__()

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
        self.TransformerLayer = self.encoderBlock(config)
        self.halt_layers = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                         nn.GELU(),
                                         nn.Linear(self.hidden_size, 1),
                                         nn.Sigmoid())

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

    def mean(self, sequence, input_mask):
        return T.sum(sequence * input_mask.unsqueeze(-1), dim=1) / T.sum(input_mask.unsqueeze(-1), dim=1)

    # %%
    def forward(self, sequence, input_mask):
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

        next_sequence = sequence
        L = min(self.layers, max(4, S))
        not_already_halted_prob = 1  # T.ones(N).float().to(sequence.device)
        accu_sequence = 0  # T.zeros(N, S, D).float(sequence.device)
        accu_penalty = T.zeros(N).float().to(sequence.device)
        total_halt_prob = T.zeros(N).float().to(sequence.device)

        for l in range(L):
            sequence0 = sequence.clone()
            accu_penalty0 = accu_penalty.clone()
            next_sequence0 = next_sequence.clone()
            sequence = self.TransformerLayer(sequence=sequence,
                                             kv_sequence=sequence,
                                             positions=positions,
                                             kv_positions=positions,
                                             query_input_mask=input_mask,
                                             input_mask=input_mask,
                                             layer_num=l + 1)["sequence"]
            # layer_num l + oned to index it from 1.
            assert sequence.size() == (N, S, D)

            mean0 = self.mean(sequence0, input_mask)
            mean1 = self.mean(sequence, input_mask)
            mean_cat = T.cat([mean0, mean1], dim=-1)
            halt_prob = self.halt_layers(mean_cat).squeeze(-1)
            current_halt_prob = not_already_halted_prob * halt_prob
            accu_sequence = (current_halt_prob.view(N, 1, 1) * sequence0) + accu_sequence
            total_halt_prob = total_halt_prob + current_halt_prob
            not_already_halted_prob = not_already_halted_prob * (1 - halt_prob)
            accu_penalty = current_halt_prob * (l + 1) + accu_penalty

            next_sequence = (1 - total_halt_prob.view(N, 1, 1)) * sequence + accu_sequence
            # print("l: {}, halt_prob: {}".format(l, halt_prob))

            update_flag = T.where(total_halt_prob >= 0.999,
                                  T.zeros(N).float().to(sequence.device),
                                  T.ones(N).float().to(sequence.device))

            accu_penalty = T.where(update_flag.bool(),
                                   accu_penalty,
                                   accu_penalty0)

            update_flag_repeat = repeat(update_flag, 'n -> n s d', n=N, s=S, d=D)

            sequence = T.where(update_flag_repeat.bool(),
                               sequence,
                               sequence0)

            next_sequence = T.where(update_flag_repeat.bool(),
                                    next_sequence,
                                    next_sequence0)

            if T.sum(update_flag) == 0.0:
                #print("halt layer: ", l)
                break

        sequence = next_sequence
        # print("accu_prob: ", accu_penalty)

        if self.config["pooling"] == "cls":
            global_state = sequence[:, 0, :]
        elif self.config["pooling"] == "cls+":
            start = sequence[:, 0, :]
            end = T.sum(sequence * END_mask, dim=1)
            global_state = self.cls_compress(T.cat([start, end], dim=-1))
        elif self.config["pooling"] == "end":
            global_state = T.sum(sequence * END_mask, dim=1)
        elif self.config["pooling"] == "mean":
            global_state = self.mean(sequence, input_mask)

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": accu_penalty}
