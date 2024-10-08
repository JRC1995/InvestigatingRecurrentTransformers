import torch as T
import numpy as np
import random
import copy


class sentence_pair_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.train = train

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


    def collate_fn(self, batch):
        copy_batch = copy.deepcopy(batch)
        sequences1_vec = [obj['sequence1_vec'] for obj in copy_batch]
        sequences2_vec = [obj['sequence2_vec'] for obj in copy_batch]
        sequences_vec = [obj["sequence_vec"] for obj in copy_batch]
        sequences1 = [obj['sequence1'] for obj in copy_batch]
        sequences2 = [obj['sequence2'] for obj in copy_batch]
        labels = [obj['label'] for obj in copy_batch]
        pairID_flag = False
        if "pairID" in batch[0]:
            pairID_flag = True
            pairIDs = [obj["pairID"] for obj in batch]

        bucket_size = len(sequences1_vec)
        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        lengths = [len(obj) for obj in sequences_vec]
        sorted_idx = np.argsort(lengths)

        sequences1_vec = self.sort_list_by_idx(sequences1_vec, sorted_idx)
        sequences2_vec = self.sort_list_by_idx(sequences2_vec, sorted_idx)
        sequences_vec = self.sort_list_by_idx(sequences_vec, sorted_idx)
        sequences1 = self.sort_list_by_idx(sequences1, sorted_idx)
        sequences2 = self.sort_list_by_idx(sequences2, sorted_idx)
        labels = self.sort_list_by_idx(labels, sorted_idx)
        if pairID_flag:
            pairIDs = self.sort_list_by_idx(pairIDs, sorted_idx)

        meta_batches = []

        i = 0
        while i < bucket_size:
            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            #max_len1 = max([len(obj) for obj in sequences1_vec[i:i + inr]])
            #max_len2 = max([len(obj) for obj in sequences2_vec[i:i + inr]])

            inr_ = inr

            j = copy.deepcopy(i)
            batches = []
            while j < i + inr:
                sequences1_vec_, input_masks1 = self.pad(sequences1_vec[j:j + inr_], PAD=self.PAD)
                sequences2_vec_, input_masks2 = self.pad(sequences2_vec[j:j + inr_], PAD=self.PAD)
                sequences_vec_, input_masks = self.pad(sequences_vec[j:j + inr_], PAD=self.PAD)

                batch = {}
                batch["sequences1_vec"] = T.tensor(sequences1_vec_).long()
                batch["sequences2_vec"] = T.tensor(sequences2_vec_).long()
                batch["sequences_vec"] = T.tensor(sequences_vec_).long()
                batch["sequences1"] = sequences1[j:j + inr_]
                batch["sequences2"] = sequences2[j:j + inr_]
                batch["labels"] = T.tensor(labels[j:j + inr_]).long()
                batch["input_masks1"] = T.tensor(input_masks1).to(T.float16)
                batch["input_masks2"] = T.tensor(input_masks2).to(T.float16)
                batch["input_masks"] = T.tensor(input_masks).to(T.float16)
                batch["batch_size"] = inr_
                if pairID_flag:
                    batch["pairIDs"] = pairIDs[j:j + inr_]
                else:
                    batch["pairIDs"] = [None] * inr_
                batches.append(batch)
                j += inr_
            i += inr

            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
