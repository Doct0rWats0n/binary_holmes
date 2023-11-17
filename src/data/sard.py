import argparse
from typing import List, Union, Dict

import os
import os.path as osp
import json

import torch
from torch.utils.data import random_split
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.data.utils import ngrams_iterator

from src.data.data_module import BaseDataModule
from training.run_experiment import DATA_CLASS_MODULE, import_class

MAX_LENGTH = 1000

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]

class SARDataset(Dataset):
    def __init__(self, raw_dir, max_length: int = MAX_LENGTH) -> None:
        super().__init__()
        tokenized_data = []
        tokenized_labels = []
        self.raw_dir = raw_dir
        self.mapping, self.reverse_mapping = self.get_mappings(self.raw_dir)
        for cwe_folder in os.listdir(self.raw_dir):
            CWE_NAME = cwe_folder.split("_")[0]
            for file in os.listdir(osp.join(self.raw_dir, cwe_folder)):
                with open(osp.join(self.raw_dir, cwe_folder, file)) as f:
                    tokenized_data.append(json.load(f))
                    tokenized_labels.append(CWE_NAME)
        self.vocab = build_vocab_from_iterator(tokenized_data, min_freq=1000, special_first=False)
        self.vocab.set_default_index(len(self.vocab))
        self.data = tokenized_data
        self.labels = tokenized_labels

        self.transforms = T.Sequential(
            T.Truncate(max_length),
            T.VocabTransform(self.vocab),
            T.ToTensor(),
            T.PadTransform(max_length, len(self.vocab))
        )
        self.label_transforms = T.Sequential(
            T.LabelToIndex(self.reverse_mapping.keys())
        )

    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        x = self.transforms(self.data[index])
        y = self.label_transforms(self.labels[index])
        y = to_categorical(y, len(self.mapping))
        return x, y
        
    @staticmethod
    def get_mappings(raw_dir) -> Union[Dict, Dict]:
        classes = []
        for cwe_folder in os.listdir(raw_dir):
            CWE_NAME = cwe_folder.split("_")[0]
            if CWE_NAME not in classes:
                classes.append(CWE_NAME)
        m =  {i : x for i, x in enumerate(classes)}
        rm = {v: k for k, v in m.items()}
        return m, rm
    
    @staticmethod
    def get_vocab(raw_dir) -> Vocab:
        tokenized_data = []
        for cwe_folder in os.listdir(raw_dir):
            for file in os.listdir(osp.join(raw_dir, cwe_folder)):
                with open(osp.join(raw_dir, cwe_folder, file)) as f:
                    tokenized_data.append(json.load(f))
        return build_vocab_from_iterator(tokenized_data, min_freq=1000)


class SARD(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.raw_dir = self.args.get("raw_dir", self.data_dirname() / "raw" / "llvmir")
        self.name_f = self.args.get("name_parsing", self._naive_name_parsing)
        self.mapping, self.reverse_mapping = SARDataset.get_mappings(self.raw_dir)
        self.input_dims = [MAX_LENGTH, ]
        self.output_dims = [len(self.mapping)]
    
    def _naive_name_parsing(self, name) -> List:
        return [], []

    def prepare_data(self, *args, **kwargs) -> None:
        print("PREPARING")
        self.dataset = SARDataset(self.raw_dir, max_length=MAX_LENGTH)
        self.vocab_size = len(self.dataset.vocab)

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