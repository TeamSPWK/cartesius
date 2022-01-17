import csv
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from torch import nn

from cartesius.data import PolygonDataModule
from cartesius.tasks import DEFAULT_TASK_W
from cartesius.tasks import TASKS
from cartesius.utils import kaggle_convert_labels


class PolygonEncoder(pl.LightningModule):
    """Class representing the full model to be trained.

    The full model is composed of :
     * Encoder
     * Task-specific heads

    Args:
        tasks (dict): Dict of Tasks to train on.
        encoder (torch.nn.Module): Encoder to train and benchmark.
        tasks_scales (list, optional): Tasks' scales (for scaling the loss appropriately). Defaults to DEFAULT_TASK_W.
        lr (float, optional): Learning rate. Defaults to 3e-4.
        kaggle_submission_file (str, optional): Path of where to save the predictions for Kaggle submission.
            Defaults to `submission.csv`.
        scheduler (str, optional): Name of the scheduler to use. For now, only `cosannwarm` is available. Pass None if
            you don't want to use any scheduler. Defaults to None.
        sched_conf (dict, optional): Arguments to pass to the scheduler's constructor, as a dict. Pass None for no
            conf. Defaults to None.
    """

    def __init__(self,
                 tasks,
                 encoder,
                 tasks_scales=DEFAULT_TASK_W.values(),
                 lr=3e-4,
                 kaggle_submission_file="submission.csv",
                 scheduler=None,
                 sched_conf=None):
        super().__init__()

        self.tasks = tasks
        self.tasks_scales = tasks_scales

        self.encoder = encoder
        self.tasks_heads = nn.ModuleList([t.get_head() for t in self.tasks.values()])

        self.learning_rate = lr
        self.lr = None

        self.kaggle_submission_file = kaggle_submission_file
        self.scheduler = scheduler
        self.sched_conf = sched_conf if sched_conf is not None else {}

    def forward(self, x):
        # Encode polygon features
        features = self.encoder(**x)

        # Extract the predictions for each task
        preds = [th(features) for th in self.tasks_heads]

        return preds

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        preds = self.forward(batch["inputs"])

        losses = []
        for (task_name, task), pred, label, s in zip(self.tasks.items(), preds, batch["labels"], self.tasks_scales):
            loss = task.get_loss_fn()(pred, label)
            self.log(f"task_losses/{task_name}", loss)
            losses.append(s * loss)  # Scale the loss

        loss = sum(losses)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        preds = self.forward(batch["inputs"])

        losses = []
        for (task_name, task), pred, label, s in zip(self.tasks.items(), preds, batch["labels"], self.tasks_scales):
            loss = task.get_loss_fn()(pred, label)
            self.log(f"val_task_losses/{task_name}", loss)
            losses.append(s * loss)  # Scale the loss

        loss = sum(losses)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        preds = self.forward(batch["inputs"])

        losses = []
        for (task_name, task), pred, label, s in zip(self.tasks.items(), preds, batch["labels"], self.tasks_scales):
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

        kaggle_rows = [kaggle_convert_labels(self.tasks.keys(), p) for p in preds]

        with open(self.kaggle_submission_file, "w", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=list(kaggle_rows[0][0].keys()))
            writer.writeheader()

            for i, kaggle_row in enumerate(kaggle_rows):
                for row in kaggle_row:
                    row["id"] = f"{i}_" + row["id"]
                    writer.writerow(row)

    def configure_optimizers(self):
        lr = self.lr or self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.scheduler == "cosannwarm":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.sched_conf)
            return [optimizer], [scheduler]
        else:
            return optimizer
