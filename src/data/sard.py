import argparse
from typing import List, Union, Dict, Tuple

import os
import os.path as osp
import json

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.data.utils import ngrams_iterator

from .chunker import Chunker

from src.data.data_module import BaseDataModule
from training.run_experiment import DATA_CLASS_MODULE, import_class
from src.models.parameters import CHUNK_SIZE, CHUNK_OVERLAP

MAX_LENGTH = 10000


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]


class SARDataset(Dataset):
    def __init__(self, raw_dir, max_length: int = MAX_LENGTH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> None:
        super().__init__()
        data_paths = []
        tokenized_labels = []
        self.raw_dir = raw_dir
        self.mapping, self.reverse_mapping = self.get_mappings(self.raw_dir)

        for index, cwe_folder in enumerate(os.listdir(self.raw_dir)):
            CWE_NAME = cwe_folder.split("_")[0]
            for file in os.listdir(osp.join(self.raw_dir, cwe_folder)):
                data_paths.append(osp.join(self.raw_dir, cwe_folder, file))
            tokenized_labels.append(CWE_NAME)
        self.vocab = self.get_vocab(raw_dir)
        self.vocab.set_default_index(len(self.vocab))
        self.data_paths = data_paths
        self.labels = tokenized_labels

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

        file_name = osp.split(self.data_paths[index])[1]
        label = file_name.split('_')[0]

        x = self.transforms(tokens_list)
        y = self.label_transforms(label)
        return x, y

    @staticmethod
    def get_mappings(raw_dir) -> tuple[dict, dict]:
        classes = []
        for cwe_folder in os.listdir(raw_dir):
            CWE_NAME = cwe_folder.split("_")[0]
            if CWE_NAME not in classes:
                classes.append(CWE_NAME)
        m = {i: x for i, x in enumerate(classes)}
        rm = {v: k for k, v in m.items()}
        return m, rm

    @staticmethod
    def get_vocab(raw_dir) -> Vocab:
        with open(osp.join(raw_dir, '../../stored_vocab.json')) as vocab_file:
            tokens = json.load(vocab_file)
            print('vocab_loaded')
        return build_vocab_from_iterator(tokens)


class SARD(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.raw_dir = self.args.get("raw_dir", self.data_dirname() / "raw" / "llvmir")
        self.name_f = self.args.get("name_parsing", self._naive_name_parsing)
        self.mapping, self.reverse_mapping = SARDataset.get_mappings(self.raw_dir)
        self.input_dims = [MAX_LENGTH, ]
        self.output_dims = [len(self.mapping)]

    def _naive_name_parsing(self, name) -> Tuple[List, List]:
        return [], []

    def prepare_data(self, *args, **kwargs) -> None:
        print("PREPARING")
        self.dataset = SARDataset(self.raw_dir, max_length=MAX_LENGTH)
        self.vocab_size = len(self.dataset.vocab)

    @staticmethod
    def collate_fn(batch):
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        ls = [b[0].shape[0] for b in batch]
        padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
        return padded, ls, torch.tensor(ys)

    def train_dataloader(self, **args) -> DataLoader:

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
        self.data_train, self.data_val = random_split(self.dataset, (train_size, len(self.dataset) - train_size))
        self.data_test = None

    def config(self):
        d = super().config()
        d["vocab_size"] = len(SARDataset.get_vocab(self.raw_dir)) + 1
        d["seq_len"] = MAX_LENGTH
        return d


if __name__ == "__main__":
    ...
