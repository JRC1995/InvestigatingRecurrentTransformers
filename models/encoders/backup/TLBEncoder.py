import torch as T
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from models.modules.TLBBlock import TLBBlock
from models.modules.GatedTLBBlock import GatedTLBBlock
from models.modules.CrossTransformerBlock import CrossTransformerBlock
from models.utils import generate_absolute_positional_embeddings


class TLBEncoder(nn.Module):
    def __init__(self, config):
        super(TLBEncoder, self).__init__()

        self.config = config
        self.update_layer_num = config["update_layers"]
        self.hidden_size = config["hidden_size"]
        self.chunk_size = config["model_chunk_size"]
        self.memory_size = config["memory_size"]

        """
        _, self.position_embeddings = generate_absolute_positional_embeddings(max_len=max(self.memory_size,
                                                                                          self.chunk_size),
                                                                              d_model=self.hidden_size)
        """

        #self.position_embeddings = nn.Embedding(self.chunk_size, self.hidden_size)
        #self.memory_position_embeddings = nn.Embedding(self.memory_size, self.hidden_size)
        #nn.init.normal_(self.position_embeddings.weight, std=0.02)
        #nn.init.normal_(self.memory_position_embeddings.weight, std=0.02)

        self.memory = nn.Parameter(T.zeros(self.memory_size, self.hidden_size).float())

        self.encoderBlock = eval(config["block"])
        self.crossBlock = CrossTransformerBlock
        self.layers = config["layers"]

        self.TLBBlocks = nn.ModuleList([self.encoderBlock(config) for _ in range(self.layers)])
        self.update_layers = nn.ModuleList([self.crossBlock(config) for _ in range(self.update_layer_num)])

        self.LN = nn.LayerNorm(self.hidden_size)
        self.eps = 1e-8

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

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

        memory = repeat(self.memory, 'm d -> n m d', n=N)
        chunked_sequence_ = []
        chunked_input_mask_ = []
        all_memories = []
        memory_mask = []
        true_chunk_nums = T.zeros(N).float().to(input_mask.device)
        #memory_positions = repeat(T.arange(0, self.memory_size).float().to(input_mask.device), 'm -> n m', n=N)
        #memory_pos_embd = self.memory_position_embeddings(memory_positions.long())
        #assert memory_pos_embd.size() == (N, self.memory_size, self.hidden_size)
        memory_input_mask = T.ones(N, self.memory_size).float().to(input_mask.device)

        for sequence, input_mask in zip(chunked_sequence, chunked_input_mask):

            N, S, D = sequence.size()
            assert input_mask.size() == (N, S)

            ones_mask = T.cat([T.ones(N, 1).float().to(input_mask.device),
                               T.zeros(N, S - 1).float().to(input_mask.device)], dim=1)
            pseudo_input_mask = ones_mask + (
                    1 - ones_mask) * input_mask  # forces one entity to be one for stability reasons
            assert pseudo_input_mask.size() == (N, S)

            positions = T.cumsum(pseudo_input_mask, dim=1) - 1

            N, S, D = sequence.size()
            original_sequence = sequence.clone()
            original_memory = memory.clone()

            # sequence = sequence + position_encodings
            # memory = memory + memory_pos

            layer_num = 0
            for l in range(self.layers):
                sequence = self.TLBBlocks[l](sequence=sequence,
                                             memory=memory,
                                             positions=positions,
                                             input_mask=pseudo_input_mask,
                                             memory_input_mask=memory_input_mask,
                                             layer_num=layer_num + 1,
                                             xpos=True)["sequence"]
                layer_num += 1

            assert sequence.size() == (N, S, D)

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

        sequence = T.cat(chunked_sequence_, dim=1)
        input_mask = T.cat(chunked_input_mask_, dim=1)
        assert sequence.size() == (N, S0, D)
        assert input_mask.size() == (N, S0)
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
                "aux_loss": None}
