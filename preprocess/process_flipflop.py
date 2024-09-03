import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import csv
import copy
import statistics

SEED = 101
dev_keys = ["mixed"]
test_keys = ["normal", "512_08", "512_01", "512_98", "1024_08", "1024_01", "1024_98"]
np.random.seed(SEED)
random.seed(SEED)
max_seq_len = 1000000

Path('../processed_data/flipflop/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/flipflop/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/flipflop/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/flipflop/test_{}.jsonl'.format(key))

metadata_save_path = fspath(Path("../processed_data/flipflop/metadata.pkl"))

labels2idx = {"0": 0, "1": 1}
vocab2idx = {"w": 0, "i": 1, "r": 2, "0": 3, "1": 4}

symbols = ["0", "1"]
ins = ["w", "r", "i"]

train_len = 160000 + 11000
test_len = 11000

unique_set = set()


def generate_date(seq_len, data_len, pi):
    start = "w"
    sequences = []
    masks = []
    p = (1 - pi) / 2
    i = 0
    while i < data_len:
        string = [start]
        mask = [0]
        state = "w"
        mem = None
        for k in range(1, seq_len):
            if state == "w":
                new_symbol = np.random.choice(symbols, size=1, p=[0.5, 0.5])[0]
                mem = new_symbol
                state = "rand"
                mask.append(0)
                string.append(new_symbol)
            elif state == "rand":
                if k == seq_len - 2:
                    new_ins = "r"
                else:
                    new_ins = np.random.choice(ins, size=1, p=[p, p, pi])[0]
                string.append(new_ins)
                mask.append(0)
                if new_ins == "w":
                    state = "w"
                elif new_ins == "i":
                    state = "i"
                elif new_ins == "r":
                    state = "r"
            elif state == "r":
                string.append(mem)
                mask.append(1)
                state = "rand"
            elif state == "i":
                new_symbol = np.random.choice(symbols, size=1, p=[0.5, 0.5])[0]
                string.append(new_symbol)
                mask.append(0)
                state = "rand"

        stringstr = "".join(string)
        if stringstr not in unique_set:
            if "r" in stringstr:
                sequences.append(string)
                masks.append(mask)
                unique_set.add(stringstr)
                #print(string)
                i += 1
                print(i)

    return sequences, masks


train_sequences_, train_masks_ = generate_date(seq_len=512, data_len=train_len, pi=0.8)
test512_08_sequences, test512_08_masks = generate_date(seq_len=512, data_len=test_len, pi=0.8)
test512_01_sequences, test512_01_masks = generate_date(seq_len=512, data_len=test_len, pi=0.1)
test512_98_sequences, test512_98_masks = generate_date(seq_len=512, data_len=test_len, pi=0.98)

test1024_08_sequences, test1024_08_masks = generate_date(seq_len=1024, data_len=test_len, pi=0.8)
test1024_01_sequences, test1024_01_masks = generate_date(seq_len=1024, data_len=test_len, pi=0.1)
test1024_98_sequences, test1024_98_masks = generate_date(seq_len=1024, data_len=test_len, pi=0.98)

dev_sequences = {}
dev_sequences["mixed"] = train_sequences_[0:1000] \
                         + test512_08_sequences[0:1000] \
                         + test512_01_sequences[0:1000] \
                         + test512_98_sequences[0:1000] \
                         + test1024_08_sequences[0:1000] \
                         + test1024_01_sequences[0:1000] \
                         + test1024_98_sequences[0:1000]

dev_masks = {}
dev_masks["mixed"] = train_masks_[0:1000]\
                         + test512_08_masks[0:1000]\
                         + test512_01_masks[0:1000]\
                         + test512_98_masks[0:1000]\
                         + test1024_08_masks[0:1000]\
                         + test1024_01_masks[0:1000]\
                         + test1024_98_masks[0:1000]
test_sequences = {}
test_masks = {}
test_sequences["normal"] = train_sequences_[1000:11000]
test_masks["normal"] = train_masks_[1000:11000]

train_sequences = train_sequences_[11000:]
train_masks = train_masks_[11000:]

test_sequences["512_08"] = test512_08_sequences[1000:]
test_masks["512_08"] = test512_08_masks[1000:]
test_sequences["512_01"] = test512_01_sequences[1000:]
test_masks["512_01"] = test512_01_masks[1000:]
test_sequences["512_98"] = test512_98_sequences[1000:]
test_masks["512_98"] = test512_98_masks[1000:]

test_sequences["1024_08"] = test1024_08_sequences[1000:]
test_masks["1024_08"] = test1024_08_masks[1000:]
test_sequences["1024_01"] = test1024_01_sequences[1000:]
test_masks["1024_01"] = test1024_01_masks[1000:]
test_sequences["1024_98"] = test1024_98_sequences[1000:]
test_masks["1024_98"] = test1024_98_masks[1000:]

def text_vectorize(text):
    return [vocab2idx[word] for word in text]


def vectorize_data(sequences, masks):
    data_dict = {}
    sequences_vec = [text_vectorize(sequence) for sequence in sequences]
    data_dict["sequence"] = sequences
    data_dict["sequence_vec"] = sequences_vec
    data_dict["mask"] = masks
    return data_dict

train_data = vectorize_data(train_sequences, train_masks)
dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences[key], dev_masks[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences[key], test_masks[key])


jsonl_save(filepath=train_save_path,
           data_dict=train_data)

for key in dev_keys:
    jsonl_save(filepath=dev_save_path[key],
               data_dict=dev_data[key])

for key in test_keys:
    jsonl_save(filepath=test_save_path[key],
               data_dict=test_data[key])

metadata = {"labels2idx": labels2idx,
            "vocab2idx": vocab2idx,
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
