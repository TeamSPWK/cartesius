import csv
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from torch import nn

from cartesius.data import PolygonDataModule
from cartesius.tasks import TASKS
from cartesius.utils import kaggle_convert_labels


class PolygonEncoder(pl.LightningModule):
    """Class representing the full model to be trained.

    The full model is composed of :
     * Encoder
     * Task-specific heads

    Args:
        conf (omegaconf.OmegaConf): Configuration.
        tasks (list): List of Tasks to train on.
        encoder (torch.nn.Module): Encoder to train and benchmark.
    """

    def __init__(self, conf, tasks, encoder):
        super().__init__()

        self.conf = conf
        self.tasks = tasks
        self.tasks_scales = conf["tasks_scales"]

        self.encoder = encoder
        self.tasks_heads = nn.ModuleList([t.get_head() for t in self.tasks])

        self.learning_rate = conf.lr
        self.lr = None

    def forward(self, x):
        # Encode polygon features
        features = self.encoder(**x)

        # Extract the predictions for each task
        preds = [th(features) for th in self.tasks_heads]

        return preds

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        labels = batch.pop("labels")

        preds = self.forward(batch)

        losses = []
        for task_name, task, pred, label, s in zip(self.conf.tasks, self.tasks, preds, labels, self.tasks_scales):
            loss = task.get_loss_fn()(pred, label)
            self.log(f"task_losses/{task_name}", loss)
            losses.append(s * loss)  # Scale the loss

        loss = sum(losses)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        labels = batch.pop("labels")
        preds = self.forward(batch)

        losses = []
        for task_name, task, pred, label, s in zip(self.conf.tasks, self.tasks, preds, labels, self.tasks_scales):
            loss = task.get_loss_fn()(pred, label)
            self.log(f"val_task_losses/{task_name}", loss)
            losses.append(s * loss)  # Scale the loss

        loss = sum(losses)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        labels = batch.pop("labels")

        preds = self.forward(batch)

        losses = []
        for task_name, task, pred, label, s in zip(self.conf.tasks, self.tasks, preds, labels, self.tasks_scales):
            loss = task.get_loss_fn()(pred, label)
            self.log(f"test_task_losses/{task_name}", loss)
            losses.append(s * loss)  # Scale the loss

        loss = sum(losses)
        self.log("test_loss", loss)
        return [p.tolist() for p in preds]

    def test_epoch_end(self, outputs):
        # Convert outputs into a list of sample, where each sample is a list of task values
        preds = []
        for output in outputs:
            # Transpose : [tasks, batch] => [batch, tasks]
            batch = list(map(list, zip(*output)))
            preds.extend(batch)

        kaggle_rows = [kaggle_convert_labels(self.conf.tasks, p) for p in preds]

        with open(self.conf.kaggle_submission_file, "w", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=list(kaggle_rows[0][0].keys()))
            writer.writeheader()

            for i, kaggle_row in enumerate(kaggle_rows):
                for row in kaggle_row:
                    row["id"] = f"{i}_" + row["id"]
                    writer.writerow(row)

    def configure_optimizers(self):
        lr = self.lr or self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.conf["scheduler"] == "cosannwarm":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=self.conf["sched_T_0"],
                                                                             T_mult=self.conf["sched_T_mult"],
                                                                             eta_min=lr *
                                                                             self.conf["sched_min_lr_ratio"])
            return [optimizer], [scheduler]
        else:
            return optimizer