import torch.nn as nn
from .FlashMHA import FlashMHA


class GatedTLBBlock(nn.Module):
    def __init__(self, config):
        super(GatedTLBBlock, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.ffn_dim = config["ffn_dim"]
        self.config = config

        self.attention = eval(config["attn_fn"])(config)
        self.cross_attention = FlashMHA(config)  # eval(config["attn_fn"])(config)

        self.drop0 = nn.Dropout(config["dropout"])
        self.drop1 = nn.Dropout(config["dropout"])
        self.LN0 = nn.LayerNorm(self.hidden_size)
        self.LN11 = nn.LayerNorm(self.hidden_size)
        self.LN12 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)

        self.ffn_gate_dim = config["ffn_gate_dim"]
        self.FFN_gate = nn.Sequential(nn.Linear(self.hidden_size, self.ffn_gate_dim),
                                      nn.Dropout(config["dropout"]),
                                      nn.GELU(),
                                      nn.Linear(self.ffn_gate_dim, self.hidden_size),
                                      nn.Dropout(config["dropout"]),
                                      nn.Sigmoid())

        self.FFN = nn.Sequential(nn.Linear(self.hidden_size, self.ffn_dim),
                                 nn.Dropout(config["dropout"]),
                                 nn.GELU(),
                                 nn.Linear(self.ffn_dim, self.hidden_size),
                                 nn.Dropout(config["dropout"]))

    """
    Forward Function
    """

    # %%
    def forward(self, sequence,
                memory,
                positions,
                input_mask,
                memory_input_mask,
                layer_num,
                kv_sequence=None,
                kv_positions=None,
                query_input_mask=None,
                position_encodings=None,
                kv_position_encodings=None,
                xpos=True):
        """
        :param sequence: batch_size x seq_len x dimensions
        :param positions: batch_size x seq_len
        :param position_encodings: batch_size x seq_len x dimensions
        :param input_mask: batch_size x seq_len
        :param layer_num: scalar (current layer number)
        :return: {sequence: batch_size x seq_len x dimensions}
        """

        N, S, D = sequence.size()

        # assert input_mask.size() == (N, S)

        res1 = sequence.clone()
        sequence = self.LN0(sequence)
        if kv_sequence is not None:
            kv_sequence = self.LN0(kv_sequence)
        sequence = self.attention(sequence=sequence,
                                  kv_sequence=kv_sequence,
                                  positions=positions,
                                  kv_positions=kv_positions,
                                  input_mask=input_mask,
                                  query_input_mask=query_input_mask,
                                  position_encodings=position_encodings,
                                  kv_position_encodings=kv_position_encodings,
                                  layer_num=layer_num,
                                  xpos=xpos)["attended_values"]
        sequence = self.drop0(sequence) + res1
        res2 = sequence.clone()

        sequence = self.LN11(sequence)
        memory = self.LN12(memory)
        sequence = self.cross_attention(sequence=sequence,
                                        kv_sequence=memory,
                                        input_mask=memory_input_mask,
                                        query_input_mask=input_mask,
                                        positions=None,
                                        kv_positions=None,
                                        layer_num=layer_num,
                                        xpos=False)["attended_values"]
        sequence = self.drop1(sequence) + res2
        res3 = sequence.clone()
        sequence = self.LN2(sequence)
        gate = self.FFN_gate(sequence)
        sequence = self.FFN(sequence) + res3
        sequence = gate * sequence + (1 - gate) * res1

        return {"sequence": sequence}
