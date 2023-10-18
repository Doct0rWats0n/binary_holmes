import argparse
import importlib

from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST
from torchvision.transforms import ToTensor

from src.data.data_module import BaseDataModule
from training.run_experiment import DATA_CLASS_MODULE, import_class


class MNIST(import_class(f"{DATA_CLASS_MODULE}.BaseDataModule")):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = MNIST.data_dirname("")
        self.transform = ToTensor();
        self.input_dims = (1, 28, 28)
        self.output_dims = (1, )
        self.mapping = list(range(10))

    def prepare_data(self, *args, **kwargs) -> None:
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [50000, 10000])
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)

