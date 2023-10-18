import argparse
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Sequence, Tuple, Union, Optional
import pytorch_lightning as pl
from pathlib import Path


SequenceOrTensor = Union[Sequence, torch.Tensor]

BATCH_SIZE = 8
NUM_WORKERS = 4


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        # Make sure to set the variables below in subclasses
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.data_train: Union[Dataset, ConcatDataset]
        self.data_val: Union[Dataset, ConcatDataset]
        self.data_test: Union[Dataset, ConcatDataset]

    @classmethod
    def data_dirname(cls, split: str = None):
        return Path(__file__).resolve().parents[2] / "data" / cls.__name__ / split if split is not None else ""

    @staticmethod
    def add_to_argparse(parser: argparse.Namespace) -> argparse.Namespace:
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        parser.add_argument("--no_train_shuffle", action="store_true", default=None)
        return parser

    def prepare_data(self) -> None:
        # donwload
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        # create self.data_train, self.data_val, self.data_test
        ...

    def train_dataloader(self, **args) -> DataLoader:
        return DataLoader(
            self.data_train,
            shuffle=not self.args.get("no_train_shuffle", False),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **args
        )

    def val_dataloader(self, **args) -> DataLoader:
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **args
        )

    def test_dataloader(self, **args) -> DataLoader:
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **args
        )

    def config(self):
        return {"input_dims": self.input_dims, "output_dims": self.output_dims, "mapping": self.mapping}