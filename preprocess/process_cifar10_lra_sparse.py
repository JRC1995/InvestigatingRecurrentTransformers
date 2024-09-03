import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import torchvision

tf = tf.compat.v1
AUTOTUNE = tf.data.experimental.AUTOTUNE

SEED = 101
dev_keys = ["normal"]
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)

train_dataset = tfds.load('cifar10', split='train[:90%]')
val_dataset = tfds.load('cifar10', split='train[90%:]')
test_dataset = tfds.load('cifar10', split='test')


def decode(x):
    decoded = {
        'inputs':
            tf.cast(tf.image.rgb_to_grayscale(x['image']), dtype=tf.int32),
        'targets':
            x['label']
    }
    return decoded


train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

Path('../processed_data/cifar10_lra_sparse').mkdir(parents=True, exist_ok=True)
train_save_path = Path('../processed_data/cifar10_lra_sparse/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/cifar10_lra_sparse/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/cifar10_lra_sparse/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/cifar10_lra_sparse/metadata.pkl"))

vocab2idx = {str(i): i for i in range(256)}
labels2idx = {str(i): i for i in range(10)}


def process_data(dataset):
    sequences = []
    labels = []

    for x in dataset:
        text = np.reshape(x["inputs"].numpy(), 32 * 32).tolist()
        label = int(x["targets"].numpy())
        sequences.append(text)
        labels.append(label)

    return sequences, labels


train_sequences, train_labels = process_data(train_dataset)

test_sequences = {}
test_labels = {}
test_sequences["normal"], test_labels["normal"] = process_data(test_dataset)

dev_sequences = {}
dev_labels = {}
dev_sequences["normal"], dev_labels["normal"] = process_data(val_dataset)


def vectorize_data(sequences, labels):
    data_dict = {}
    data_dict["sequence"] = [["..."] for _ in sequences]
    data_dict["sequence_vec"] = sequences
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences, train_labels)

dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences[key], test_labels[key])

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
