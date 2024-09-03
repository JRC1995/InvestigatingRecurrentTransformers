import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from models.modules import *


class UniversalTransformerEncoder_last(nn.Module):
    def __init__(self, config):
        super(UniversalTransformerEncoder_last, self).__init__()

        self.config = config
        self.layers = config["layers"]
        self.hidden_size = config["hidden_size"]

        self.START = nn.Parameter(T.randn(self.hidden_size))
        self.END = nn.Parameter(T.randn(self.hidden_size))

        self.penalty_gamma = config["penalty_gamma"]
        self.attn_based_halt = config["attn_based_halt"]
        self.transition_based_halt = config["transition_based_halt"]
        self.global_halt = config["global_halt"]
        self.mean_based_halt = config["mean_based_halt"]
        self.causal = config["causal"]
        self.ex_halt = config["ex_halt"]
        self.head_dim = config["head_dim"]

        if self.ex_halt:
            self.HALT = nn.Parameter(T.randn(self.head_dim))

        self.encoderBlock = eval(config["block"])
        self.TransformerLayer = self.encoderBlock(config)

        if self.transition_based_halt:
            factor = 2
        else:
            factor = 1

        if self.attn_based_halt:
            self.cat_compress = nn.Sequential(nn.Linear(factor * self.hidden_size, self.hidden_size),
                                              nn.GELU())
            self.attn_scorer = nn.Linear(self.hidden_size, 1)
            self.halt_layers = nn.Sequential(nn.Linear(self.hidden_size, 1),
                                             nn.Sigmoid())
        else:
            if self.ex_halt:
                self.halt_layers = nn.Sequential(nn.Linear(factor * self.head_dim, self.head_dim),
                                                 nn.GELU(),
                                                 nn.Linear(self.head_dim, 1),
                                                 nn.Sigmoid())
            else:
                self.halt_layers = nn.Sequential(nn.Linear(factor * self.hidden_size, self.hidden_size),
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
        return T.sum(sequence * input_mask, dim=1) / T.sum(input_mask, dim=1)

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
        if self.causal:
            raise ValueError("Not Implemented")

        sequence, input_mask, END_mask = self.augment_sequence(sequence, input_mask)
        N, S, D = sequence.size()
        positions = T.cumsum(input_mask, dim=1) - 1

        START = repeat(self.START, 'd -> n s d', n=N, s=1)
        END = repeat(self.END, 'd -> n s d', n=N, s=1)

        START_mask = T.cat([T.ones(N, 1, 1).float().to(input_mask.device),
                            T.zeros(N, S - 1, 1).float().to(input_mask.device)], dim=1)

        last_mask = T.cat([END_mask[:, 1:, :],
                           T.zeros(N, 1, 1).float().to(input_mask.device)], dim=1)

        next_sequence = sequence
        L = min(self.layers, S)
        not_already_halted_prob = 1  # T.ones(N).float().to(sequence.device)
        accu_sequence = 0  # T.zeros(N, S, D).float(sequence.device)
        if self.global_halt:
            accu_penalty = T.zeros(N).float().to(sequence.device)
            accu_penalty_slack = T.zeros(N).float().to(sequence.device)
            total_halt_prob = T.zeros(N).float().to(sequence.device)
        else:
            accu_penalty = T.zeros(N, S).float().to(sequence.device)
            accu_penalty_slack = T.zeros(N, S).float().to(sequence.device)
            total_halt_prob = T.zeros(N, S).float().to(sequence.device)
        if self.ex_halt:
            halt_sequence = repeat(self.HALT, 'd -> n s d', d=self.head_dim, n=N, s=S)
            halt_sequence_original = halt_sequence.clone()
        else:
            halt_sequence = None
            halt_sequence_original = None

        for l in range(L):
            # print("L: ", l)
            sequence0 = sequence.clone()
            accu_penalty0 = accu_penalty.clone()
            accu_penalty_slack0 = accu_penalty_slack.clone()
            next_sequence0 = next_sequence.clone()
            if self.ex_halt:
                halt_sequence0 = halt_sequence.clone()

            kv_sequence = sequence if self.global_halt else next_sequence

            sequence_dict = self.TransformerLayer(sequence=sequence,
                                                  halt_sequence=halt_sequence_original,
                                                  kv_sequence=kv_sequence,
                                                  positions=positions,
                                                  kv_positions=positions,
                                                  query_input_mask=input_mask,
                                                  input_mask=input_mask,
                                                  layer_num=l + 1)
            sequence, halt_sequence = sequence_dict["sequence"], sequence_dict["halt_sequence"]

            sequence = START_mask * START + (1 - START_mask) * sequence
            sequence = END_mask * END + (1 - END_mask) * sequence

            # layer_num l + oned to index it from 1.
            assert sequence.size() == (N, S, D)
            if self.ex_halt:
                assert halt_sequence.size() == (N, S, self.head_dim)

            if self.attn_based_halt:
                if self.transition_based_halt:
                    cat_sequence = self.cat_compress(T.cat([sequence0, sequence], dim=-1))
                else:
                    cat_sequence = self.cat_compress(sequence0)
                attn = F.softmax(self.attn_scorer(cat_sequence).squeeze(-1) * input_mask + (1 - input_mask) * -99999,
                                 dim=1)
                attn = attn.unsqueeze(-1)
                attn_cont = T.sum(attn * cat_sequence, dim=1)
                halt_prob = self.halt_layers(attn_cont).squeeze(-1)
            elif self.mean_based_halt:
                global_state0 = self.mean(sequence0, input_mask.unsqueeze(-1))
                global_state1 = self.mean(sequence, input_mask.unsqueeze(-1))
                if self.transition_based_halt:
                    halt_prob = self.halt_layers(T.cat([global_state0, global_state1], dim=-1)).squeeze(-1)
                else:
                    halt_prob = self.halt_layers(global_state0).squeeze(-1)
            else:
                if self.ex_halt:
                    if self.transition_based_halt:
                        halt_prob = self.halt_layers(T.cat([halt_sequence0, halt_sequence], dim=-1)).squeeze(-1)
                    else:
                        halt_prob = self.halt_layers(halt_sequence).squeeze(-1)
                else:
                    if self.transition_based_halt:
                        halt_prob = self.halt_layers(T.cat([sequence0, sequence], dim=-1)).squeeze(-1)
                    else:
                        halt_prob = self.halt_layers(sequence0).squeeze(-1)

                halt_prob = START_mask.squeeze(-1) + (1 - START_mask).squeeze(-1) * halt_prob
                halt_prob = END_mask.squeeze(-1) + (1 - END_mask).squeeze(-1) * halt_prob

            X = 1 if self.global_halt else S

            current_halt_prob = not_already_halted_prob * halt_prob
            accu_sequence = (current_halt_prob.view(N, X, 1) * sequence0) + accu_sequence
            total_halt_prob = total_halt_prob + current_halt_prob
            not_already_halted_prob = not_already_halted_prob * (1 - halt_prob)

            accu_penalty = current_halt_prob * (l + 1) + accu_penalty
            accu_penalty_slack = (1 - total_halt_prob) * (l + 1)

            next_sequence = (1 - total_halt_prob.view(N, X, 1)) * sequence + accu_sequence
            # print("l: {}, halt_prob: {}".format(l, halt_prob))

            if self.global_halt:
                update_flag = T.where(total_halt_prob >= 0.999,
                                      T.zeros(N).float().to(sequence.device),
                                      T.ones(N).float().to(sequence.device))
            else:
                update_flag = T.where(total_halt_prob >= 0.999,
                                      T.zeros(N, S).float().to(sequence.device),
                                      T.ones(N, S).float().to(sequence.device)) * input_mask.view(N, S)

            accu_penalty = T.where(update_flag.bool(),
                                   accu_penalty,
                                   accu_penalty0)

            # print("accu_penalty0: ", accu_penalty)

            accu_penalty_slack = T.where(update_flag.bool(),
                                         accu_penalty_slack,
                                         accu_penalty_slack0)

            # print("accu_penalty_slack: ", accu_penalty_slack)

            if self.global_halt:
                update_flag_repeat = repeat(update_flag, 'n -> n s d', n=N, s=S, d=D)
            else:
                update_flag_repeat = repeat(update_flag, 'n s -> n s d', n=N, s=S, d=D)

            sequence = T.where(update_flag_repeat.bool(),
                               sequence,
                               sequence0)

            next_sequence = T.where(update_flag_repeat.bool(),
                                    next_sequence,
                                    next_sequence0)

            if self.ex_halt:
                update_flag_repeat = repeat(update_flag, 'n s -> n s d', n=N, s=S, d=self.head_dim)
                halt_sequence = T.where(update_flag_repeat.bool(),
                                        halt_sequence,
                                        halt_sequence0)

            # print("accu_prob1: ", accu_penalty)

            if T.sum(update_flag) == 0.0:
                # print("halt layer: ", l)
                break

        sequence = next_sequence
        # print("accu_prob10: ", accu_penalty)
        accu_penalty = accu_penalty + accu_penalty_slack
        # print("accu_prob1: ", accu_penalty)
        if not self.global_halt:
            accu_penalty = self.mean(accu_penalty, input_mask)

        if self.config["pooling"] == "cls":
            global_state = sequence[:, 0, :]
        elif self.config["pooling"] == "cls+":
            start = sequence[:, 0, :]
            end = T.sum(sequence * END_mask, dim=1)
            global_state = self.cls_compress(T.cat([start, end], dim=-1))
        elif self.config["pooling"] == "end":
            global_state = T.sum(sequence * last_mask, dim=1)
        elif self.config["pooling"] == "mean":
            global_state = self.mean(sequence, input_mask.unsqueeze(-1))

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": self.penalty_gamma * accu_penalty}
