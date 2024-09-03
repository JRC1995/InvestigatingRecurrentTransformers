import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from models.modules import *


class RMTEncoder(nn.Module):
    def __init__(self, config):
        super(RMTEncoder, self).__init__()

        self.config = config
        self.layers = config["layers"]
        self.hidden_size = config["hidden_size"]
        self.chunk_size = config["model_chunk_size"]
        self.memory_size = config["memory_size"]

        self.memory_init = nn.Parameter(T.zeros(self.memory_size, self.hidden_size))
        self.memory_collectors = nn.Parameter(T.zeros(self.memory_size, self.hidden_size))

        self.position_embeddings = nn.Embedding(self.chunk_size, self.hidden_size)
        self.memory1_position_embeddings = nn.Embedding(self.memory_size, self.hidden_size)
        self.memory2_position_embeddings = nn.Embedding(self.memory_size, self.hidden_size)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        nn.init.normal_(self.memory1_position_embeddings.weight, std=0.02)
        nn.init.normal_(self.memory2_position_embeddings.weight, std=0.02)

        """
        self.leaf_transform = config["leaf_transform"]

        if self.leaf_transform:
            self.leaf_transform_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                      nn.LayerNorm(self.hidden_size))
        self.position_embeddings = nn.Embedding(self.chunk_size + (self.memory_size * 2),
                                                self.hidden_size)
        T.nn.init.normal_(self.position_embeddings.weight, std=0.02)
        """

        self.encoderBlock = eval(config["block"])
        self.TransformerLayers = nn.ModuleList([self.encoderBlock(config) for _ in range(self.layers)])

        self.eps = 1e-8

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%
    def augment_sequence(self, memory, sequence, input_mask):
        N, S, D = sequence.size()
        assert input_mask.size() == (N, S)

        """
        AUGMENT SEQUENCE WITH START AND END TOKENS
        """
        if memory is None:
            memory = repeat(self.memory_init, 'm d -> n m d', n=N)

        assert memory.size() == (N, self.memory_size, D)
        sequence = T.cat([memory, sequence], dim=1)
        assert sequence.size() == (N, S + self.memory_size, D)
        input_mask = T.cat([T.ones(N, self.memory_size).to(input_mask.dtype).to(input_mask.device),
                            input_mask], dim=1)
        assert input_mask.size() == (N, S + self.memory_size)
        S = S + self.memory_size

        memory_collectors = repeat(self.memory_collectors, 'm d -> n m d', n=N)
        assert memory_collectors.size() == (N, self.memory_size, D)
        sequence = T.cat([sequence, memory_collectors], dim=1)
        assert sequence.size() == (N, S + self.memory_size, D)
        input_mask = T.cat([input_mask,
                            T.ones(N, self.memory_size).to(input_mask.dtype).to(input_mask.device)], dim=1)
        assert input_mask.size() == (N, S + self.memory_size)

        return sequence, input_mask

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
        N, S0, _ = sequence.size()

        chunk_num = math.ceil(S0 / self.chunk_size)

        chunked_sequence = T.chunk(sequence, chunks=chunk_num, dim=1)
        assert len(chunked_sequence) == chunk_num
        chunked_input_mask = T.chunk(input_mask, chunks=chunk_num, dim=1)

        memory = None
        chunked_sequence_ = []
        chunked_input_mask_ = []
        all_memories = []
        memory_mask = []
        true_chunk_nums = T.zeros(N).float().to(input_mask.device)

        memory_positions = repeat(T.arange(0, self.memory_size).float().to(input_mask.device), 'm -> n m', n=N)
        mem_pos_enc1 = self.memory1_position_embeddings(memory_positions.long())
        mem_pos_enc2 = self.memory2_position_embeddings(memory_positions.long())

        for chunk_sequence, chunk_input_mask in zip(chunked_sequence, chunked_input_mask):
            chunk_size = chunk_sequence.size(1)
            chunk_positions = repeat(T.arange(0, self.chunk_size).float().to(input_mask.device), 'c -> n c', n=N)
            #position_encodings = T.zeros(N, chunk_input_mask.size(1), self.hidden_size).float().to(mem_pos_enc2.device)
            chunk_position_encodings = self.position_embeddings(chunk_positions.long())
            position_encodings = T.cat([mem_pos_enc1,
                                        chunk_position_encodings[:, 0:chunk_size, :],
                                        mem_pos_enc2], dim=1)

            sequence, input_mask = self.augment_sequence(memory=memory,
                                                         sequence=chunk_sequence,
                                                         input_mask=chunk_input_mask)
            positions = T.cumsum(input_mask, dim=1) - 1
            N, S, D = sequence.size()
           # position_encodings = self.position_embeddings(positions.long())
            assert position_encodings.size() == (N, S, self.hidden_size)

            N, S, D = sequence.size()
            original_sequence = sequence.clone()

            for l in range(self.layers):
                sequence = self.TransformerLayers[l](sequence=sequence,
                                                     positions=positions,
                                                     input_mask=input_mask,
                                                     position_encodings=position_encodings,
                                                     layer_num=l + 1)["sequence"]
                # layer_num l + oned to index it from 1.
                assert sequence.size() == (N, S, D)

            gate = input_mask[:, 0].view(N, 1, 1)
            sequence = gate * sequence + (1 - gate) * original_sequence
            true_chunk_nums = true_chunk_nums + input_mask[:, 0]

            chunked_sequence_.append(sequence[:, self.memory_size:-self.memory_size, :])
            chunked_input_mask_.append(input_mask[:, self.memory_size:-self.memory_size])

            memory = sequence[:, -self.memory_size:, :]
            assert memory.size() == (N, self.memory_size, D)
            all_memories.append(memory)
            memory_mask.append(input_mask[:, 0])

        sequence = T.cat(chunked_sequence_, dim=1)
        input_mask = T.cat(chunked_input_mask_, dim=1)
        assert sequence.size() == (N, S0, D)
        assert input_mask.size() == (N, S0)
        all_memories = T.cat(all_memories, dim=1)
        assert all_memories.size() == (N, chunk_num * self.memory_size, D)
        memory_mask = T.stack(memory_mask, dim=1)
        assert memory_mask.size() == (N, chunk_num)
        memory_mask = repeat(memory_mask, 'n c -> n (c m) d', m=self.memory_size, d=1)

        if self.config["pooling"] in ["cls", "cls+"]:
            global_state = memory[:, -1, :]
        elif self.config["pooling"] == "mean":
            global_state = T.mean(memory, dim=1)
            # global_state = T.sum(all_memories * memory_mask, dim=1) / (true_chunk_nums.view(N, 1) * self.memory_size)

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": None}
