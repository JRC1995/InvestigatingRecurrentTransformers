import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *


class seqlabel_framework(nn.Module):
    def __init__(self, data, config):

        super(seqlabel_framework, self).__init__()

        self.config = config
        self.classes_num = data["classes_num"] if data["classes_num"] > 2 else 1
        self.pad_id = data["PAD_id"]
        self.hidden_size = config["hidden_size"]
        self.dropout = config["dropout"]

        if config["embedding_type"] == "sparse":
            vocab_len = data["vocab_len"]
            self.word_embedding = nn.Embedding(vocab_len, config["embd_dim"], padding_idx=self.pad_id)
            T.nn.init.normal_(self.word_embedding.weight, std=0.02)
        else:
            self.word_embedding = nn.Linear(1, config["embd_dim"])

        encoder_fn = eval(config["encoder_type"])
        self.encoder = encoder_fn(config)

        self.classifier_block = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], self.classes_num)
        )

    # %%
    def forward(self, batch):

        sequence = batch["sequences_vec"]
        input_mask = batch["input_masks"]
        N, S = sequence.size()
        # EMBEDDING BLOCK
        sequence = self.word_embedding(sequence)
        sequence = sequence.to(T.float16)
        sequence = F.dropout(sequence, p=self.dropout, training=self.training)

        sequence_dict = self.encoder(sequence, input_mask)
        sequence = sequence_dict["sequence"]

        aux_loss = None
        if "aux_loss" in sequence_dict:
            aux_loss = sequence_dict["aux_loss"]
            if aux_loss is not None:
                aux_loss = aux_loss.mean()

        logits = self.classifier_block(sequence)

        assert logits.size() == (N, S, self.classes_num)
        return {"logits": logits, "aux_loss": aux_loss}
