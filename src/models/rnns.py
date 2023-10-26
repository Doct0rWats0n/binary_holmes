import argparse
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 128
LSTM_DIM = 64
FC_DIM = 32
FC_DROPOUT = 0.5
LEAKY_COEF = 0.3
LSTM_NUM_LAYERS = 10
BIDIR = False

class BLSTM(nn.Module):
    """Quick port of [BLSTM](https://github.com/johnb110/VDPython/blob/master/blstm.py)"""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        input_dim = self.data_config["vocab_size"]
        seq_len = self.data_config["seq_len"]
        num_classes = len(self.data_config["mapping"])


        self.embed = nn.Embedding(input_dim, EMBED_DIM)
        self.bilstm = nn.LSTM(EMBED_DIM, LSTM_DIM, num_layers=LSTM_NUM_LAYERS, batch_first=True, bidirectional=BIDIR)
        self.bilstm2 = nn.LSTM(LSTM_DIM, 1, num_layers=LSTM_NUM_LAYERS, batch_first=True, bidirectional=BIDIR)
        self.fc1 = nn.Linear(LSTM_DIM * seq_len * (2 if BIDIR else 1), FC_DIM)
        # self.drop1 = nn.Dropout(FC_DROPOUT)
        self.fc2 = nn.Linear(FC_DIM, FC_DIM)
        # self.drop2 = nn.Dropout(FC_DROPOUT)
        self.fc3 = nn.Linear(FC_DIM, num_classes)
    
    def forward(self, x):
        x = self.embed(x)
        x, (hs, hc) = self.bilstm(x)
        # x, (hs, hc) = self.bilstm2(x, (hs, hc))
        # x = torch.cat((hs[-2, :, :], hs[-1, :, :]), dim=1)
        x = x.reshape(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), LEAKY_COEF)
        # x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), LEAKY_COEF)
        # x = self.drop2(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--edim", type=int, default=EMBED_DIM)
        parser.add_argument("--lstm", type=int, default=LSTM_DIM)
        parser.add_argument("--fc", type=int, default=FC_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        parser.add_argument("--leaky", type=float, default=LEAKY_COEF)
        return parser
