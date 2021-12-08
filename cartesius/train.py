import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from torch import nn

from cartesius.data import PolygonDataModule
from cartesius.models import create_model
from cartesius.tasks import TASKS
from cartesius.utils import create_tags
from cartesius.utils import load_conf


class PolygonEncoder(pl.LightningModule):
    """Class representing the full model to be trained.

    The full model is composed of :
     * Encoder
     * Task-specific heads

    Args:
        conf (omegaconf.OmegaConf): Configuration.
        tasks (list): List of Tasks to train on.
    """

    def __init__(self, conf, tasks):
        super().__init__()

        self.conf = conf
        self.tasks = tasks
        self.tasks_scales = conf["tasks_scales"]

        self.encoder = create_model(conf.model_name, conf)
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
        return loss

    def configure_optimizers(self):
        lr = self.lr or self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.conf["scheduler"] == "cosannwarm":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.conf["sched_T_0"], T_mult=self.conf["sched_T_mult"], eta_min=lr * self.conf["sched_min_lr_ratio"])
            return [optimizer], [scheduler]
        else:
            return optimizer


def main():
    """Entry point of Cartesius for general training.
    This function reads the configuration provided and run the training + testing.
    """
    conf = load_conf()

    # If no seed is set, generate one, and set the seed
    if not conf.seed:
        conf.seed = random.randrange(1000)
    pl.seed_everything(conf.seed, workers=True)

    # Resolve tasks
    tasks = [TASKS[t](conf) for t in conf.tasks]

    model = PolygonEncoder(conf, tasks)
    data = PolygonDataModule(conf, tasks)

    wandb_logger = pl.loggers.WandbLogger(project=conf.project_name, config=conf, tags=create_tags(conf))
    if conf.watch_model:
        wandb_logger.watch(model, log="all")
    mc = ModelCheckpoint(monitor="val_loss", mode="min", filename="{step}-{val_loss:.4f}")
    trainer = pl.Trainer(
        gpus=1,
        logger=wandb_logger,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True), mc],
        gradient_clip_val=conf.grad_clip,
        max_time=conf.max_time,
        auto_lr_find=conf.auto_lr_find,
        default_root_dir=conf.save_dir,
        num_sanity_val_steps=-1,
    )

    if conf.train:
        trainer.tune(model, datamodule=data)
        trainer.fit(model, datamodule=data)

        # Update the config to record the best checkpoint
        wandb_logger.experiment.config["best_ckpt"] = mc.best_model_path

    if conf.test:
        ckpt = conf.ckpt or mc.best_model_path
        trainer.test(model, datamodule=data, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
