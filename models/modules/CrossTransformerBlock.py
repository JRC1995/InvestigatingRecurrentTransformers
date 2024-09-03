import torch.nn as nn
from .FlashMHA import FlashMHA


class CrossTransformerBlock(nn.Module):
    def __init__(self, config):
        super(CrossTransformerBlock, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.ffn_dim = config["ffn_dim"]
        self.config = config

        self.attention = FlashMHA(config) #eval(config["attn_fn"])(config)
        self.drop = nn.Dropout(config["dropout"])
        self.LN11 = nn.LayerNorm(self.hidden_size)
        self.LN12 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)

        self.FFN = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Dropout(config["dropout"]),
                                 nn.GELU(),
                                 nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Dropout(config["dropout"]))

    """
    Forward Function
    """

    # %%
    def forward(self, sequence,
                kv_sequence,
                positions,
                kv_positions,
                input_mask,
                query_input_mask,
                layer_num,
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

        #assert input_mask.size() == (N, S)

        res1 = sequence.clone()
        sequence = self.LN11(sequence)
        kv_sequence = self.LN12(kv_sequence)
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
        sequence = self.drop(sequence) + res1
        res2 = sequence.clone()
        sequence = self.FFN(self.LN2(sequence)) + res2

        return {"sequence": sequence}
