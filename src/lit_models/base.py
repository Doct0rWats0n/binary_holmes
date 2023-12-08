import argparse

import lightning.pytorch as pl
import torch
from torchmetrics import Accuracy

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.data_config = self.model.data_config
        self.mapping = self.data_config["mapping"]
        self.input_dims = self.data_config["input_dims"]

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if loss not in ("transformer",):
            self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy(task="multiclass", num_classes=len(self.mapping))
        self.val_acc = Accuracy(task="multiclass", num_classes=len(self.mapping))
        self.test_acc = Accuracy(task="multiclass", num_classes=len(self.mapping))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}

    def forward(self, x, l):
        return self.model(x, l)

    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.train_acc(logits, y)
        # self.val_acc(y, torch.nn.functional.softmax(logits, dim=1).argmax(dim=1))

        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        outputs = {"loss": loss}
        self.add_on_first_batch({"logits": logits.detach()}, outputs, batch_idx)

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        x, l, y = batch
        # print("\n" * 100, x.shape, "\n" * 100)
        logits = self(x, l)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = self.loss_fn(logits, y)

        return x, y, logits, loss

    def validation_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        #print(y, "\n", logits,  "\n" * 10)
        # self.val_acc(torch.nn.functional.softmax(logits, dim=1).argmax(dim=0), y.argmax(dim=0))
        #print(torch.nn.functional.softmax(logits, dim=1).shape)
        # self.val_acc(y, torch.nn.functional.softmax(logits, dim=1).argmax(dim=1))
        # print(logits.shape, y.argmax(dim=-1).shape, y.shape)
        self.val_acc(logits, y)

        self.log("validation_loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}
        self.add_on_first_batch({"logits": logits.detach()}, outputs, batch_idx)

        return outputs

    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.test_acc(logits, y.argmax(dim=1))

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def add_on_first_batch(self, metrics, outputs, batch_idx):
        if batch_idx == 0:
            outputs.update(metrics)

    def add_on_logged_batches(self, metrics, outputs):
        if self.is_logged_batch:
            outputs.update(metrics)

    def is_logged_batch(self):
        if self.trainer is None:
            return False
        else:
            return self.trainer._logger_connector.should_update_logs
