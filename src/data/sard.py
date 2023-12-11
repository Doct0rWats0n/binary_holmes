import argparse
from typing import List, Union, Dict, Tuple

import os
import os.path as osp
import ujson as json
from collections import OrderedDict, Counter

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.data.utils import ngrams_iterator

from .chunker import Chunker

from src.data.data_module import BaseDataModule
from training.run_experiment import DATA_CLASS_MODULE, import_class
from src.models.parameters import CHUNK_SIZE, CHUNK_OVERLAP

MAX_LENGTH = 13000
GOOD_LABEL = "GOOD"

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]


class SARDataset(Dataset):
    def __init__(self, raw_dir, max_length: int = MAX_LENGTH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, include_good=True) -> None:
        super().__init__()
        data_paths = []
        self.raw_dir = raw_dir
        self.mapping, self.reverse_mapping = self.get_mappings(self.raw_dir, include_good=include_good)
        labels = []
        for _, cwe_folder in enumerate(os.listdir(self.raw_dir)):
            splitted_folder = cwe_folder.split("_")
            cwe = splitted_folder[0]
            is_good = GOOD_LABEL.lower() in splitted_folder[-1].lower()
            if not include_good and is_good:
                continue
            is_good = is_good and include_good
            for file in os.listdir(osp.join(self.raw_dir, cwe_folder)):
                data_paths.append(osp.join(self.raw_dir, cwe_folder, file))
                labels.append(cwe if not is_good else GOOD_LABEL)
        
        self.vocab = self.get_vocab(raw_dir)
        self.vocab.set_default_index(len(self.vocab))
        self.data_paths = data_paths
        self.labels = labels
        self.include_good = include_good

        self.transforms = T.Sequential(
            T.Truncate(max_length),
            T.VocabTransform(self.vocab),
            T.ToTensor(),
            # T.PadTransform(max_length, len(self.vocab))
        )
        self.label_transforms = T.Sequential(
            T.LabelToIndex(self.reverse_mapping.keys())
        )

        self.chunker = Chunker(chunk_size, chunk_overlap)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        with open(self.data_paths[index]) as json_file:
            tokens = json.load(json_file)
            tokens_list = [token[0] for token in tokens]
        label = self.labels[index]
        x = self.transforms(tokens_list)
        y = self.label_transforms(label)
        return x, y

    @staticmethod
    def get_mappings(raw_dir, include_good=True) -> tuple[dict, dict]:
        classes = [GOOD_LABEL] if include_good else []
        for cwe_folder in os.listdir(raw_dir):
            CWE_NAME = cwe_folder.split("_")[0]
            if CWE_NAME not in classes:
                classes.append(CWE_NAME)
        rm = OrderedDict()
        for i, v in enumerate(classes):
            rm[v] = i
        return classes, rm

    @staticmethod
    def get_vocab(raw_dir) -> Vocab:
        with open(osp.join(raw_dir, '../../stored_vocab.json')) as vocab_file:
            tokens = json.load(vocab_file)
            print('vocab_loaded')
        return build_vocab_from_iterator(tokens)

class SARD_CHUNK(Dataset):
    def __init__(self, data_paths, labels, DATASET: SARDataset) -> None:
        super().__init__()
        self.data_paths = data_paths
        self.labels = labels
        self.transforms = DATASET.transforms
        self.label_transforms = DATASET.label_transforms
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        with open(self.data_paths[index]) as json_file:
            tokens = json.load(json_file)
            tokens_list = [token[0] for token in tokens]
        label = self.labels[index]
        x = self.transforms(tokens_list)
        y = self.label_transforms(label)
        return x, y

class SARD(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.include_good = not self.args.get("ignore_good", False)
        self.raw_dir = self.args.get("raw_dir", self.data_dirname() / "raw" / "llvmir")
        self.name_f = self.args.get("name_parsing", self._naive_name_parsing)
        self.mapping, self.reverse_mapping = SARDataset.get_mappings(self.raw_dir, self.include_good)
        self.input_dims = [MAX_LENGTH, ]
        self.output_dims = [len(self.mapping)]

    def _naive_name_parsing(self, name) -> Tuple[List, List]:
        return [], []

    def prepare_data(self, *args, **kwargs) -> None:
        print("PREPARING")
        self.dataset = SARDataset(self.raw_dir, max_length=MAX_LENGTH, include_good=self.include_good)
        self.vocab_size = len(self.dataset.vocab)

    @staticmethod
    def collate_fn(batch):
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        ls = [b[0].shape[0] for b in batch]
        padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
        return padded, ls, torch.tensor(ys)

    def train_dataloader(self, **args) -> DataLoader:
        if self.include_good:
            counter = Counter(self.data_train.labels)
            weights = torch.tensor([1 / counter[y] for y in self.data_train.labels], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, len(self.data_train.labels))
            return DataLoader(
                self.data_train,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
                **args
            )
        return DataLoader(
            self.data_train,
            collate_fn=self.collate_fn,
            shuffle=not self.args.get("no_train_shuffle", False),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **args
        )

    def val_dataloader(self, **args) -> DataLoader:
        return DataLoader(
            self.data_val,
            collate_fn=self.collate_fn,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **args
        )

    def setup(self, stage=None) -> None:
        train_size = int(len(self.dataset) * 0.8)
        if self.include_good:
            # ALL THIS SHIT BECAUSE OF random_split and GOOD SAMPLES
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(self.dataset.data_paths, self.dataset.labels, stratify=self.dataset.labels)
            self.data_train = SARD_CHUNK(X_train, y_train, self.dataset)
            self.data_val = SARD_CHUNK(X_val, y_val, self.dataset)
        else:
            self.data_train, self.data_val = random_split(self.dataset, (train_size, len(self.dataset) - train_size))
        self.data_test = None

    def config(self):
        d = super().config()
        d["vocab_size"] = len(SARDataset.get_vocab(self.raw_dir)) + 1
        d["seq_len"] = MAX_LENGTH
        return d
    
    @staticmethod
    def add_to_argparse(parser: argparse.Namespace) -> argparse.Namespace:
        parser = super(SARD, SARD).add_to_argparse(parser)
        parser.add_argument("--ignore_good", action="store_true")
        return parser


if __name__ == "__main__":
    ...
