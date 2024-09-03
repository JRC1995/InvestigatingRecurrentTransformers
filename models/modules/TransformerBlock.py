import torch.nn as nn
from .FlashMHA import FlashMHA
from .AMHA import AMHA
from .MHA import MHA
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.ffn_dim = config["ffn_dim"]
        self.config = config

        self.attention = eval(config["attn_fn"])(config)
        self.drop = nn.Dropout(config["dropout"])
        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.head_dim = config["head_dim"]
        if self.config["ex_halt"]:
            self.HLN1 = nn.LayerNorm(config["head_dim"])
            self.HLN2 = nn.LayerNorm(config["head_dim"])
            self.HFFN = nn.Sequential(nn.Linear(self.head_dim, 2 * self.head_dim),
                                      nn.Dropout(config["dropout"]),
                                      nn.GELU(),
                                      nn.Linear(2 * self.head_dim, self.head_dim),
                                      nn.Dropout(config["dropout"]))
            nn.init.xavier_uniform_(self.HFFN[0].weight)
            nn.init.xavier_uniform_(self.HFFN[3].weight)
            nn.init.normal_(self.HFFN[0].bias, std=1e-6)
            nn.init.normal_(self.HFFN[3].bias, std=1e-6)

        self.FFN = nn.Sequential(nn.Linear(self.hidden_size, self.ffn_dim),
                                 nn.Dropout(config["dropout"]),
                                 nn.GELU(),
                                 nn.Linear(self.ffn_dim, self.hidden_size),
                                 nn.Dropout(config["dropout"]))

        nn.init.xavier_uniform_(self.FFN[0].weight)
        nn.init.xavier_uniform_(self.FFN[3].weight)
        nn.init.normal_(self.FFN[0].bias, std=1e-6)
        nn.init.normal_(self.FFN[3].bias, std=1e-6)

    """
    Forward Function
    """

    # %%
    def forward(self, sequence, kv_sequence, positions, kv_positions,
                input_mask, query_input_mask, layer_num,
                halt_sequence=None, position_encodings=None,
                halt_position_encodings=None, kv_position_encodings=None, xpos=True):
        """
        :param sequence: batch_size x seq_len x dimensions
        :param kv_sequence: batch_size x kv_seq_len x dimensions
        :param positions: batch_size x seq_len
        :param kv_positions: batch_size x kv_seq_len
        :param position_encodings: batch_size x seq_len x dimensions
        :param kv_position_encodings: batch_size x kv_seq_len x dimensions
        :param input_mask: batch_size x kv_seq_len
        :param query_input_mask: batch_size x seq_len
        :param layer_num: scalar (current layer number)
        :return: {sequence: batch_size x seq_len x dimensions}
        """

        N, S, D = sequence.size()
        assert input_mask.size() == (N, S)

        res1 = sequence.clone()
        sequence = self.LN1(sequence)
        if kv_sequence is not None:
            kv_sequence = self.LN1(kv_sequence)

        if halt_sequence is not None:
            halt_sequence = self.HLN1(halt_sequence)

        sequence_dict = self.attention(sequence=sequence,
                                       halt_sequence=halt_sequence,
                                       kv_sequence=kv_sequence,
                                       positions=positions,
                                       kv_positions=kv_positions,
                                       input_mask=input_mask,
                                       query_input_mask=query_input_mask,
                                       position_encodings=position_encodings,
                                       halt_position_encodings=halt_position_encodings,
                                       kv_position_encodings=kv_position_encodings,
                                       layer_num=layer_num,
                                       xpos=xpos)

        sequence = sequence_dict["attended_values"]
        sequence = self.drop(sequence) + res1
        res2 = sequence.clone()
        sequence = self.FFN(self.LN2(sequence)) + res2

        if halt_sequence is not None:
            halt_sequence = sequence_dict["halt_sequence"]
            halt_sequence = F.dropout(halt_sequence, p=self.config["dropout"], training=self.training)
            hres2 = halt_sequence.clone()
            halt_sequence = self.HFFN(self.HLN2(halt_sequence)) + hres2

        return {"sequence": sequence, "halt_sequence": halt_sequence}
