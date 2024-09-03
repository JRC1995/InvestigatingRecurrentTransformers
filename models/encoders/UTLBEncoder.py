import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from models.modules.TLBBlock import TLBBlock
from models.modules.GatedTLBBlock import GatedTLBBlock
from models.modules.CrossTransformerBlock import CrossTransformerBlock


class UTLBEncoder(nn.Module):
    def __init__(self, config):
        super(UTLBEncoder, self).__init__()

        self.config = config
        self.update_layer_num = config["update_layers"]
        self.hidden_size = config["hidden_size"]
        self.chunk_size = config["model_chunk_size"]
        self.memory_size = config["memory_size"]
        self.penalty_gamma = config["penalty_gamma"]
        self.attn_based_halt = config["attn_based_halt"]
        self.transition_based_halt = config["transition_based_halt"]
        self.global_halt = config["global_halt"]
        self.mean_based_halt = config["mean_based_halt"]
        self.causal = config["causal"]

        self.memory = nn.Parameter(T.zeros(self.memory_size, self.hidden_size).float())

        self.encoderBlock = eval(config["block"])
        self.crossBlock = CrossTransformerBlock
        self.layers = config["layers"]

        self.TLBBlocks = self.encoderBlock(config)
        self.update_layers = nn.ModuleList([self.crossBlock(config) for _ in range(self.update_layer_num)])

        self.LN = nn.LayerNorm(self.hidden_size)
        self.eps = 1e-8

        self.penalty_gamma = config["penalty_gamma"]
        self.attn_based_halt = config["attn_based_halt"]
        self.transition_based_halt = config["transition_based_halt"]
        self.global_halt = config["global_halt"]

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
            self.halt_layers = nn.Sequential(nn.Linear(factor * self.hidden_size, self.hidden_size),
                                             nn.GELU(),
                                             nn.Linear(self.hidden_size, 1),
                                             nn.Sigmoid())

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

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
        """
        if self.causal:
            raise ValueError("Not Implemented")
        """
        N, S0, _ = sequence.size()

        chunk_num = math.ceil(S0 / self.chunk_size)

        chunked_sequence = T.chunk(sequence, chunks=chunk_num, dim=1)
        assert len(chunked_sequence) == chunk_num
        chunked_input_mask = T.chunk(input_mask, chunks=chunk_num, dim=1)

        memory = repeat(self.memory, 'm d -> n m d', n=N)
        chunked_sequence_ = []
        chunked_input_mask_ = []
        all_memories = []
        memory_mask = []
        true_chunk_nums = T.zeros(N).float().to(input_mask.device)
        memory_input_mask = T.ones(N, self.memory_size).float().to(input_mask.device)
        accu_accu_penalty = 0

        if "global_position_encoding" in self.config:
            global_encoding = self.config["global_position_encoding"]
        else:
            global_encoding = False

        if global_encoding:
            positions = T.cumsum(input_mask, dim=1) - 1
            chunked_positions = T.chunk(positions, chunks=chunk_num, dim=1)

        ci = 0


        for sequence, input_mask in zip(chunked_sequence, chunked_input_mask):

            N, S, D = sequence.size()
            assert input_mask.size() == (N, S)

            ones_mask = T.cat([T.ones(N, 1).float().to(input_mask.device),
                               T.zeros(N, S - 1).float().to(input_mask.device)], dim=1)
            pseudo_input_mask = ones_mask + (
                    1 - ones_mask) * input_mask  # forces one entity to be one for stability reasons
            assert pseudo_input_mask.size() == (N, S)

            if self.causal:
                denom = T.cumsum(pseudo_input_mask, dim=1).view(N, S, 1)

            if global_encoding:
                positions = chunked_positions[ci]
            else:
                positions = T.cumsum(pseudo_input_mask, dim=1) - 1

            N, S, D = sequence.size()
            original_sequence = sequence.clone()
            original_memory = memory.clone()

            # sequence = sequence + position_encodings
            # memory = memory + memory_pos

            next_sequence = sequence
            L = min(self.layers, S+1)
            lengths = T.sum(input_mask, dim=1)
            not_already_halted_prob = 1  # T.ones(N).float().to(sequence.device)
            accu_sequence = 0  # T.zeros(N, S, D).float(sequence.device)

            if self.global_halt and not self.causal:
                accu_penalty = T.zeros(N).float().to(sequence.device)
                accu_penalty_slack = T.zeros(N).float().to(sequence.device)
                total_halt_prob = T.zeros(N).float().to(sequence.device)
            else:
                accu_penalty = T.zeros(N, S).float().to(sequence.device)
                accu_penalty_slack = T.zeros(N, S).float().to(sequence.device)
                total_halt_prob = T.zeros(N, S).float().to(sequence.device)

            layer_num = 0
            for l in range(L):
                # print("L: ", l)
                sequence0 = sequence.clone()
                accu_penalty0 = accu_penalty.clone()
                accu_penalty_slack0 = accu_penalty_slack.clone()
                next_sequence0 = next_sequence.clone()

                kv_sequence = sequence if self.global_halt else next_sequence

                sequence = self.TLBBlocks(sequence=sequence,
                                          kv_sequence=kv_sequence,
                                          memory=memory,
                                          positions=positions,
                                          kv_positions=positions,
                                          query_input_mask=pseudo_input_mask,
                                          input_mask=pseudo_input_mask,
                                          memory_input_mask=memory_input_mask,
                                          layer_num=layer_num + 1,
                                          xpos=True)["sequence"]
                assert sequence.size() == (N, S, D)

                force_halt = T.where(l >= lengths + 1,
                                     T.ones(N).float().to(sequence.device),
                                     T.zeros(N).float().to(sequence.device))

                if self.attn_based_halt:
                    if self.causal:
                        raise ValueError("Not implemented")
                    if self.transition_based_halt:
                        cat_sequence = self.cat_compress(T.cat([sequence0, sequence], dim=-1))
                    else:
                        cat_sequence = self.cat_compress(sequence0)
                    attn = F.softmax(self.attn_scorer(cat_sequence).squeeze(-1) * pseudo_input_mask \
                                     + (1 - pseudo_input_mask) * -99999, dim=1)
                    attn = attn.unsqueeze(-1)
                    attn_cont = T.sum(attn * cat_sequence, dim=1)
                    halt_prob = self.halt_layers(attn_cont).squeeze(-1)
                elif self.mean_based_halt:
                    if self.causal:
                        global_state0 = (T.cumsum(sequence0, dim=1) * pseudo_input_mask.unsqueeze(-1)) / denom
                        global_state1 = (T.cumsum(sequence, dim=1) * pseudo_input_mask.unsqueeze(-1)) / denom
                    else:
                        global_state0 = self.mean(sequence0, pseudo_input_mask.unsqueeze(-1))
                        global_state1 = self.mean(sequence, pseudo_input_mask.unsqueeze(-1))
                    if self.transition_based_halt:
                        halt_prob = self.halt_layers(T.cat([global_state0, global_state1], dim=-1)).squeeze(-1)
                    else:
                        halt_prob = self.halt_layers(global_state0).squeeze(-1)
                else:
                    if self.transition_based_halt:
                        halt_prob = self.halt_layers(T.cat([sequence0, sequence], dim=-1)).squeeze(-1)
                    else:
                        halt_prob = self.halt_layers(sequence0).squeeze(-1)

                X = 1 if (self.global_halt and not self.causal) else S

                current_halt_prob = not_already_halted_prob * halt_prob
                accu_sequence = (current_halt_prob.view(N, X, 1) * sequence0) + accu_sequence
                total_halt_prob = total_halt_prob + current_halt_prob
                not_already_halted_prob = not_already_halted_prob * (1 - halt_prob)

                accu_penalty = current_halt_prob * (l + 1) + accu_penalty
                accu_penalty_slack = (1 - total_halt_prob) * (l + 1)

                next_sequence = (1 - total_halt_prob.view(N, X, 1)) * sequence + accu_sequence
                # print("l: {}, halt_prob: {}".format(l, halt_prob))

                if self.global_halt and not self.causal:
                    update_flag = T.where(total_halt_prob >= 0.999,
                                          T.zeros(N).float().to(sequence.device),
                                          T.ones(N).float().to(sequence.device))
                    update_flag = update_flag * (1 - force_halt)
                else:
                    update_flag = T.where(total_halt_prob >= 0.999,
                                          T.zeros(N, S).float().to(sequence.device),
                                          T.ones(N, S).float().to(sequence.device)) * input_mask.view(N, S)
                    update_flag = update_flag * (1 - force_halt.unsqueeze(-1))

                accu_penalty = T.where(update_flag.bool(),
                                       accu_penalty,
                                       accu_penalty0)

                # print("accu_penalty0: ", accu_penalty)

                accu_penalty_slack = T.where(update_flag.bool(),
                                             accu_penalty_slack,
                                             accu_penalty_slack0)

                # print("accu_penalty_slack: ", accu_penalty_slack)

                if self.global_halt and not self.causal:
                    update_flag_repeat = repeat(update_flag, 'n -> n s d', n=N, s=S, d=D)
                else:
                    update_flag_repeat = repeat(update_flag, 'n s -> n s d', n=N, s=S, d=D)

                sequence = T.where(update_flag_repeat.bool(),
                                   sequence,
                                   sequence0)

                next_sequence = T.where(update_flag_repeat.bool(),
                                        next_sequence,
                                        next_sequence0)

                # print("accu_prob1: ", accu_penalty)
                layer_num += 1

                if T.sum(update_flag) == 0.0:
                    # print("halt layer: ", l)
                    break



            assert sequence.size() == (N, S, D)
            sequence = next_sequence
            # print("accu_prob10: ", accu_penalty)
            accu_penalty = accu_penalty + accu_penalty_slack
            # print("accu_prob1: ", accu_penalty)
            if (not self.global_halt) or self.causal:
                accu_penalty = self.mean(accu_penalty, pseudo_input_mask)
            accu_accu_penalty = accu_accu_penalty + (accu_penalty * input_mask[:, 0])

            for l in range(self.update_layer_num):
                memory = self.update_layers[l](sequence=memory,
                                               kv_sequence=sequence,
                                               positions=None,
                                               kv_positions=None,
                                               input_mask=pseudo_input_mask,
                                               query_input_mask=memory_input_mask,
                                               layer_num=10000,
                                               xpos=False)["sequence"]
                assert memory.size() == (N, self.memory_size, D)

            gate = input_mask[:, 0]
            gate1 = repeat(gate, 'n -> n s d', s=S, d=D)
            gate2 = repeat(gate, 'n -> n m d', m=self.memory_size, d=D)

            sequence = T.where(gate1 == 1,
                               sequence,
                               original_sequence)
            memory = T.where(gate2 == 1,
                             memory,
                             original_memory)

            # sequence = gate * sequence + (1 - gate) * original_sequence
            # memory = gate * memory + (1 - gate) * memory
            true_chunk_nums = true_chunk_nums + input_mask[:, 0]

            chunked_sequence_.append(sequence)
            chunked_input_mask_.append(input_mask)

            assert memory.size() == (N, self.memory_size, D)
            all_memories.append(self.LN(memory))
            memory_mask.append(input_mask[:, 0])
            ci += 1

        sequence = T.cat(chunked_sequence_, dim=1)
        input_mask = T.cat(chunked_input_mask_, dim=1)
        assert sequence.size() == (N, S0, D)
        assert input_mask.size() == (N, S0)
        accu_penalty = accu_accu_penalty / true_chunk_nums
        # all_memories = T.cat(all_memories, dim=1)
        # assert all_memories.size() == (N, chunk_num * self.memory_size, D)
        # memory_mask = T.stack(memory_mask, dim=1)
        # assert memory_mask.size() == (N, chunk_num)
        # memory_mask = repeat(memory_mask, 'n c -> n (c m) d', m=self.memory_size, d=1)

        if self.config["pooling"] in ["cls", "cls+", "end"]:
            global_state = all_memories[-1][:, -1, :]
        elif self.config["pooling"] == "mean":
            global_state = T.mean(all_memories[-1], dim=1)
            # global_state = T.sum(all_memories * memory_mask, dim=1) / (true_chunk_nums.view(N, 1) * self.memory_size)

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": self.penalty_gamma * accu_penalty}
