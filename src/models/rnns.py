import argparse
from typing import Any, Dict

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .parameters import *


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
        num_classes = len(self.data_config["mapping"])
        e_dim = self.args.get("embed")
        lstm_dim = self.args.get("lstm")
        n_lstms = self.args.get("n_lstm")
        fc_dim = self.args.get("fc")
        dc_dropout_dim = self.args.get("fc_dropout")
        self.leaky = self.args.get("leaky")

        self.embed = nn.Embedding(input_dim, e_dim)
        self.rnn = nn.LSTM(e_dim, lstm_dim, num_layers=n_lstms, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dc_dropout_dim)
        self.fc1 = nn.Linear(lstm_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

        self.h = [torch.zeros(n_lstms, self.args["batch_size"], lstm_dim).cuda() for _ in range(2)]

    def forward_words(self, x):
        raw, h = self.rnn(self.embed(x), self.h)
        dropped = self.drop1(raw)
        x = F.leaky_relu(self.fc1(dropped), self.leaky)
        x = self.fc2(x)
        self.h = [h_.detach() for h_ in h]
        return x, raw, dropped

    def forward_cat(self, x, l):
        pack = nn.utils.rnn.pack_padded_sequence(self.embed(x), l, batch_first=True, enforce_sorted=False)
        raw, (hs, hc) = self.rnn(pack)
        x = torch.cat((hs[-2,:,:], hs[-1,:,:]), dim=1)
        dropped = self.drop1(x)
        x = F.leaky_relu(self.fc1(dropped), self.leaky)
        x = self.fc2(x)
        return x

    def forward(self, x, l):
        return self.forward_cat(x, l)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--embed", type=int, default=lstm_params["EMBED_DIM"])
        parser.add_argument("--lstm", type=int, default=lstm_params["LSTM_DIM"])
        parser.add_argument("--n_lstm", type=int, default=lstm_params["LSTM_NUM_LAYERS"])
        parser.add_argument("--fc", type=int, default=lstm_params["FC_DIM"])
        parser.add_argument("--fc_dropout", type=float, default=lstm_params["FC_DROPOUT"])
        parser.add_argument("--leaky", type=float, default=lstm_params["LEAKY_COEF"])
        return parser


class GRU(nn.Module):

    def __init__(
            self,
            data_config: Dict[str, Any],
            args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        input_dim = self.data_config["vocab_size"]
        num_classes = len(self.data_config["mapping"])
        e_dim = self.args.get("embed", gru_params["EMBED_DIM"])
        gru_dim = self.args.get("gru", gru_params["GRU_DIM"])
        n_gru = self.args.get("n_gru", gru_params["GRU_NUM_LAYERS"])
        fc_dim = self.args.get("fc", gru_params["FC_DIM"])
        dc_dropout_dim = self.args.get("fc_dropout", gru_params["FC_DROPOUT"])
        self.leaky = self.args.get("leaky", gru_params["LEAKY_COEF"])

        self.embed = nn.Embedding(input_dim, e_dim)
        self.rnn = nn.GRU(e_dim, gru_dim, num_layers=n_gru, batch_first=True, bidirectional=False)
        self.rnn_2 = nn.GRU(gru_dim, 1, num_layers=1, batch_first=True, bidirectional=False)
        # self.drop1 = nn.Dropout(dc_dropout_dim)
        self.fc1 = nn.Linear(gru_dim * 2, fc_dim)
        self.fc2 = nn.Linear(CHUNK_SIZE, num_classes)
        self.fc = nn.Linear(gru_dim, fc_dim)

        self.h = [torch.zeros(n_gru, self.data_config.get("batch_size"), gru_dim).cuda() for _ in range(2)]

    def forward_words(self, x):
        raw, h = self.rnn(self.embed(x), self.h)
        dropped = self.drop1(raw)
        x = F.leaky_relu(self.fc1(dropped), self.leaky)
        x = self.fc2(x)
        self.h = [h_.detach() for h_ in h]
        return x, raw, dropped

    # def forward_cat(self, x):
    #     raw, hs = self.rnn(self.embed(x))
    #     x = torch.cat((hs[-2, :], hs[-1, :]), dim=-1)
    #     dropped = self.drop1(x)
    #     x = F.leaky_relu(self.fc1(dropped), self.leaky)
    #     x = self.fc2(x)
    #     return x

    def modified_forward(self, x):
        assert x.shape[0] == 1
        embedded = self.embed(x)
        h = torch.zeros(1, self.data_config.get("batch_size"),
                        self.args.get("gru", gru_params["GRU_DIM"])).to(x.device)
        num_seq = x.shape[1]
        for i in range(num_seq):
            embedded_one_seq = embedded[:, i]
            grued_one_seq, h = self.rnn(embedded_one_seq, h)
        squeezed_mini_batch, h_ = self.rnn_2(grued_one_seq)
        squeezed_mini_batch = squeezed_mini_batch.squeeze(-1)
        res = self.fc2(squeezed_mini_batch)
        return res

    def forward(self, x):
        return self.modified_forward(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--embed", type=int, default=gru_params["EMBED_DIM"])
        parser.add_argument("--gru", type=int, default=gru_params["GRU_DIM"])
        parser.add_argument("--n_gru", type=int, default=gru_params["GRU_NUM_LAYERS"])
        parser.add_argument("--fc", type=int, default=gru_params["FC_DIM"])
        parser.add_argument("--fc_dropout", type=float, default=gru_params["FC_DROPOUT"])
        parser.add_argument("--leaky", type=float, default=gru_params["LEAKY_COEF"])
        return parser
