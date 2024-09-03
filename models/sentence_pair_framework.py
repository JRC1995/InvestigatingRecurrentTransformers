import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *

class sentence_pair_framework(nn.Module):
    def __init__(self, data, config):

        super(sentence_pair_framework, self).__init__()

        self.config = config
        self.classes_num = data["classes_num"] if data["classes_num"] > 2 else 1 # we do sigmoid when "classes_num" == 2
        self.pad_id = data["PAD_id"]
        self.unk_id = data["UNK_id"]
        self.hidden_size = config["hidden_size"]
        self.dropout = config["dropout"]

        if config["embedding_type"] == "sparse":
            vocab_len = data["vocab_len"]
            self.word_embedding = nn.Embedding(vocab_len, config["embd_dim"], padding_idx=self.pad_id)
            T.nn.init.normal_(self.word_embedding.weight, std=0.02)
        else:
            self.word_embedding = nn.Linear(1, config["embd_dim"])

        if config["embd_dim"] != config["hidden_size"]:
            self.embd_transform = nn.Linear(config["embd_dim"], config["hidden_size"])

        encoder_fn = eval(config["encoder_type"])
        self.encoder = encoder_fn(config)

        self.classifier_block = nn.Sequential(
            nn.Linear(4 * config["hidden_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], self.classes_num)
        )

    # %%
    def forward(self, batch):

        sequence1 = batch["sequences1_vec"]
        sequence2 = batch["sequences2_vec"]
        input_mask1 = batch["input_masks1"]
        input_mask2 = batch["input_masks2"]

        N = sequence1.size(0)

        # EMBEDDING BLOCK
        sequence1 = self.word_embedding(sequence1)
        sequence2 = self.word_embedding(sequence2)

        if self.config["embd_dim"] != self.config["hidden_size"]:
            sequence1 = self.embd_transform(sequence1)
            sequence2 = self.embd_transform(sequence2)

        sequence1 = F.dropout(sequence1, p=self.dropout, training=self.training)
        sequence2 = F.dropout(sequence2, p=self.dropout, training=self.training)

        if "batch_pair" in self.config and self.config["batch_pair"]:
            pad = T.zeros(N, 1, self.hidden_size).float().to(sequence1.device)
            zero = T.zeros(N, 1).float().to(sequence1.device)

            max_s = max(sequence1.size(1), sequence2.size(1))

            while sequence1.size(1) < max_s:
                sequence1 = T.cat([sequence1, pad.clone()], dim=1)
                input_mask1 = T.cat([input_mask1, zero.clone()], dim=1)

            while sequence2.size(1) < max_s:
                sequence2 = T.cat([sequence2, pad.clone()], dim=1)
                input_mask2 = T.cat([input_mask2, zero.clone()], dim=1)

            sequence = T.cat([sequence1, sequence2], dim=0)
            input_mask = T.cat([input_mask1, input_mask2], dim=0)
            sequence_dict = self.encoder(sequence, input_mask)

            sequence1_dict = {}
            sequence2_dict = {}
            for key in sequence_dict:
                if sequence_dict[key] is None:
                    sequence1_dict[key] = None
                    sequence2_dict[key] = None
                else:
                    sequence1_dict[key] = sequence_dict[key][0:N]
                    sequence2_dict[key] = sequence_dict[key][N:]

        else:
            # ENCODER BLOCK
            sequence1_dict = self.encoder(sequence1, input_mask1)
            sequence2_dict = self.encoder(sequence2, input_mask2)

        aux_loss = None
        if "aux_loss" in sequence1_dict:
            aux_loss1 = sequence1_dict["aux_loss"]
            aux_loss2 = sequence2_dict["aux_loss"]
            if aux_loss1 is not None and aux_loss2 is not None:
                aux_loss = (aux_loss1.mean() + aux_loss2.mean()) / 2
                #aux_loss = aux_loss.mean()

        feats1 = sequence1_dict["global_state"]
        feats2 = sequence2_dict["global_state"]

        feats = T.cat([feats1, feats2,
                       feats1 * feats2,
                       T.abs(feats1 - feats2)], dim=-1)

        logits = self.classifier_block(feats)
        assert logits.size() == (N, self.classes_num)

        return {"logits": logits,
                "aux_loss": aux_loss}


