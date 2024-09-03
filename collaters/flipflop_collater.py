import torch as T
import numpy as np
import random
import copy

class flipflop_collater:
    def __init__(self, PAD, config, train):
        self.PAD = 1
        self.SOS = 1
        self.config = config
        self.train = train
        self.labels2idx = self.config["labels2idx"]

    def pad(self, items, PAD):
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def sort_list_by_idx(self, objs, idx):
        return [objs[i] for i in idx]

    def cond(self, l, m):
        if m == 0:
            return 0
        else:
            return self.labels2idx[l]

    def label_transform(self, labels, loss_masks):
        labels_ = []
        for label, loss_mask in zip(labels, loss_masks):
            label_ = [self.cond(l, m) for l, m in zip(label, loss_mask)]
            labels_.append(label_)
        return labels_

    def seq_transform(self, sequences):
        return [([self.SOS] + sequence[:-1]) for sequence in sequences]

    def collate_fn(self, batch):
        copy_batch = copy.deepcopy(batch)
        sequences_vec = [obj['sequence_vec'] for obj in copy_batch]
        sequences = [obj['sequence'] for obj in copy_batch]
        labels = copy.deepcopy(sequences)
        loss_masks = [obj['mask'] for obj in copy_batch]
        labels = self.label_transform(labels, loss_masks)
        sequences_vec = self.seq_transform(sequences_vec)

        bucket_size = len(sequences_vec)
        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        lengths = [len(obj) for obj in sequences_vec]
        sorted_idx = np.argsort(lengths)

        sequences_vec = self.sort_list_by_idx(sequences_vec, sorted_idx)
        sequences = self.sort_list_by_idx(sequences, sorted_idx)
        labels = self.sort_list_by_idx(labels, sorted_idx)
        loss_masks = self.sort_list_by_idx(loss_masks, sorted_idx)

        meta_batches = []
        i = 0
        while i < bucket_size:
            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            max_len = max([len(obj) for obj in sequences_vec[i:i + inr]])
            inr_ = inr

            j = copy.deepcopy(i)
            batches = []
            while j < i + inr:
                batch = {}
                sequences_vec_, input_masks = self.pad(sequences_vec[j:j + inr_], PAD=self.PAD)
                labels_, _ = self.pad(labels[j:j + inr_], PAD=0)
                loss_masks_, _ = self.pad(loss_masks[j:j + inr_], PAD=0)
                batch["sequences_vec"] = T.tensor(sequences_vec_).long()
                batch["sequences"] = sequences[j:j+inr_]
                batch["labels"] = T.tensor(labels_).long()
                batch["loss_masks"] = T.tensor(loss_masks_).float()
                batch["input_masks"] = T.tensor(input_masks).to(T.float16)
                batch["batch_size"] = inr_
                batches.append(batch)
                j += inr_
            i += inr

            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
