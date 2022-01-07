from matplotlib import pyplot as plt
from numpy import ma
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
import torch
from tqdm import tqdm
import wandb

from cartesius.data import PolygonDataModule
from cartesius.tasks import TASKS
from cartesius.tokenizers import TOKENIZERS


class CustomCallback(Callback):
    """Base class for custom callback
    """

    def __init__(self, conf, tasks):
        super().__init__()
        self.conf = conf
        self.tasks = tasks
        self.tasks_scales = conf["tasks_scales"]


class TransformerHardSamplePlotCallback(CustomCallback):
    """Custom callback that plots hard examples
    """
    @staticmethod
    def plot_one(subplots, df, i, j, col):
        ax = subplots[i][j]
        ax.axis("off")
        ax.set_aspect("equal")
        ax.plot(*np.swapaxes(df[col][i], 0, 1))

    @classmethod
    def plot_batch(cls, df, title, projection):
        fig = plt.figure(figsize=[2 * x for x in df.shape])
        fig.suptitle(title, fontsize=16)
        subplots = fig.subplots(*df.shape, subplot_kw={"projection": projection})
        cols = df.columns
        for ax, col in zip(subplots[0], cols):
            ax.set_title(col)
        df.reset_index(drop=True, inplace=True)
        for i in df.index:
            for j, col in enumerate(df.columns):
                cls.plot_one(subplots, df, i, j, col)
        wandb.log({f"{title}": wandb.Image(plt)})
        plt.close()

    @staticmethod
    def poly_coords_from_feature(x):
        coords = ma.array(x["polygon"], mask=np.tile(np.logical_not(x["mask"]), (2,1)).T).compressed().reshape(-1, 2)
        # if coords[0] != coords[-1]:
        #     coords = np.concatenate([coords, [coords[0]]])
        return coords

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):  # pylint: disable=unused-argument
        labels = batch.pop("labels")
        preds = pl_module(batch)
        batch["labels"] = labels
        hard_sample_table = pd.DataFrame(columns=self.conf.tasks)
        val_loss_table = pd.DataFrame(columns=["polygon"] + list(self.conf.tasks))
        val_loss_table["polygon"] = list(batch["polygon"].cpu().numpy())
        val_loss_table["mask"] = list(batch["mask"].cpu().numpy())
        for task_name, task, pred, label, s in zip(self.conf.tasks, self.tasks, preds, labels, self.tasks_scales):
            loss = task.get_loss_fn()(pred, label, reduction="none")
            if len(loss.shape) > 1:
                loss = torch.mean(loss, 1)
            val_loss_table[task_name] = list((s * loss).cpu().numpy())
            top_n = val_loss_table.nlargest(self.conf.hard_n, task_name)
            top_n["polygon"] = top_n.apply(self.poly_coords_from_feature, axis="columns")
            hard_sample_table[task_name] = top_n["polygon"].reset_index(drop=True)
        polar_tokenizers = []
        projection = None if trainer.datamodule.tokenizer not in polar_tokenizers else "polar"
        self.plot_batch(hard_sample_table, f"Top {self.conf.hard_n} Hard Samples", projection)


class LabelHistogramPlotCallback(CustomCallback):
    """Custom callback plots label distribution as histogram
    """

    def __init__(self, conf, tasks):
        super().__init__(conf, tasks)
        tasks = [TASKS[t](conf) for t in conf.tasks]
        tokenizer = TOKENIZERS[conf["tokenizer"]](**conf)
        data = PolygonDataModule(conf, tasks, tokenizer)
        data.setup()
        for data_split_type in ["train", "val", "test"]:
            setattr(self, f"{data_split_type}_data", getattr(data, f"{data_split_type}_dataloader")())

    def get_batch(self, data_split_type):
        return next(iter(getattr(self, f"{data_split_type}_data")))

    def append_batch_label(self, data_split_type):
        batch = self.get_batch(data_split_type)
        labels = pd.DataFrame(columns=self.conf.tasks)
        for i, label_name in enumerate(self.conf.tasks):
            labels[label_name] = batch["labels"][i]
        iterable_task_names = {}
        for label_, task_name in zip(labels.iloc[0], self.conf.tasks):
            try:
                iterable_task_names[task_name] = len(label_)
            except TypeError:
                pass

        for k, v in iterable_task_names.items():
            labels[[f"{k}_{v_}" for v_ in range(v)]] = pd.DataFrame(labels[k].tolist(), index=labels.index)
            del labels[k]

        if hasattr(self, f"{data_split_type}_table"):
            labels = pd.concat([getattr(self, f"{data_split_type}_table"), labels])
        setattr(self, f"{data_split_type}_table", labels)
        setattr(getattr(self, f"{data_split_type}_table"), "data_split_type", data_split_type)

        return getattr(self, f"{data_split_type}_table")

    @staticmethod
    def update_histogram(table):
        for task_name in table.columns:
            wandb.log({
                f"{table.data_split_type}_{task_name}":
                    wandb.plot.histogram(wandb.Table(dataframe=table),
                                         task_name,
                                         title=f"Label Histogram ({table.data_split_type} : {task_name})")
            })

    def on_fit_start(self, trainer, pl_module):  # pylint: disable=unused-argument
        print("Reading dataset to generate histogram...")
        for data_split_type in ["train", "val", "test"]:
            for _ in tqdm(range(self.conf.n_batch_per_epoch), desc=f"{data_split_type}_batch"):
                label_table = self.append_batch_label(data_split_type)
            self.update_histogram(label_table)


CALLBACKS = {
    "transformer_hard_sample_plot": TransformerHardSamplePlotCallback,
    "label_histogram_plot": LabelHistogramPlotCallback
}
