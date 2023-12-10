import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace, ArgumentParser
from typing import Dict, Any

class CNN1D(nn.Module):
    def __init__(self,
        data_config: Dict[str, Any],
        args: Namespace = None,) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        input_dim = self.data_config["vocab_size"]
        num_classes = len(self.data_config["mapping"])
        self.emded = nn.Embedding(input_dim, 8)
        self.layers = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=120, stride=60),
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=30, stride=10),
            nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc1 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(32, num_classes)
        )
            
    def forward(self, x, _):
        e = self.emded(x).transpose(1, 2)
        cnned = self.layers(e).squeeze()
        return self.fc1(cnned)

    @staticmethod
    def add_to_argparse(parser):
        return parser