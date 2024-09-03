import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
from PIL import Image
import math
import torchvision
from einops.layers.torch import Rearrange, Reduce

SEED = 42
dev_keys = ["normal"]
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)

data_dir = "../data/pathfinder32"

# There's an empty file in the dataset
blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

vocab2idx = {str(i): i for i in range(0, 256)}
labels2idx = {"path_exist": 1, "path_does_not_exist": 0}

Path('../processed_data/pathfinder_lra_sparse').mkdir(parents=True, exist_ok=True)
train_save_path = Path('../processed_data/pathfinder_lra_sparse/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/pathfinder_lra_sparse/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/pathfinder_lra_sparse/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/pathfinder_lra_sparse/metadata.pkl"))


data_dir = Path(data_dir).expanduser()
assert data_dir.is_dir(), f"data_dir {str(data_dir)} does not exist"


def default_transforms():
    pool = 1
    tokenize = True
    center = False
    sequential = True
    transform_list = [torchvision.transforms.ToTensor()]
    if pool > 1:
        transform_list.append(
            Reduce(
                "1 (h h2) (w w2) -> 1 h w",
                "mean",
                h2=pool,
                w2=pool,
            )
        )
    if tokenize:
        transform_list.append(
            torchvision.transforms.Lambda(lambda x: (x * 255).long())
        )
    else:
        if center:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
    if sequential:
        # If tokenize, it makes more sense to get rid of the channel dimension
        transform_list.append(
            Rearrange("1 h w -> (h w)")
            if tokenize
            else Rearrange("1 h w -> (h w) 1")
        )
    else:
        transform_list.append(Rearrange("1 h w -> h w 1"))
    return torchvision.transforms.Compose(transform_list)

# self.transform = transform
samples = []
# for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
for diff_level in ["curv_contour_length_14"]:
    path_list = sorted(
        list((data_dir / diff_level / "metadata").glob("*.npy")),
        key=lambda path: int(path.stem),
    )
    assert path_list, "No metadata found"
    for metadata_file in path_list:
        with open(metadata_file, "r") as f:
            for metadata in f.read().splitlines():
                metadata = metadata.split()
                image_path = Path(diff_level) / metadata[0] / metadata[1]
                print(str(Path(data_dir.stem) / image_path))
                if (
                        str(Path(data_dir.stem) / image_path)
                        not in blacklist
                ):
                    label = int(metadata[3])
                    samples.append((image_path, label))

sequences = []
labels = []
transform_fn = default_transforms()
for sample in samples:
    path, target = sample
    with open(data_dir / path, "rb") as f:
        sample = Image.open(f).convert("L")  # Open in grayscale
    sample = transform_fn(sample)
    #print("sample 0: ", sample.size())
    text = sample.numpy().reshape((32 * 32)).tolist()
    print("text: ", text)
    sequences.append(text)
    labels.append(target)
    #print("sample:", sample.numpy().reshape((32 * 32)).tolist())
    #print("target: ", target)

idx = [id for id in range(len(sequences))]
random.shuffle(idx)
sequences = [sequences[i] for i in idx]
labels = [labels[i] for i in idx]
test_len = math.ceil(0.1*len(sequences))
dev_len = test_len

test_sequences = {}
test_labels = {}
test_sequences["normal"], test_labels["normal"] = sequences[0:test_len], labels[0:test_len]
dev_sequences = {}
dev_labels = {}
dev_sequences["normal"], dev_labels["normal"] = sequences[test_len:test_len+dev_len], labels[test_len:test_len+dev_len]
train_sequences, train_labels = sequences[test_len+dev_len:], labels[test_len+dev_len:]


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
