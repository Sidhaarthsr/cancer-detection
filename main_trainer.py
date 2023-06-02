import os
import datetime

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .train_test_epoch import train_epoch, test_epoch, progress_bar


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device,
        save_dir,
        visualizer,
        model_saving_frequency: int = 1,
        model_name_prefix: str = "model",
        data_getter: Callable = itemgetter("image"),
        target_getter: Callable = itemgetter("target"),
        stage_progress: bool = True,
        get_key_metric: Callable = itemgetter("top1")
    ):
        
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_saving_frequency = model_saving_frequency
        self.model_name_prefix = self.model_name_prefix
        self.save_dir = save_dir
        self.visualizer = visualizer
        self.data_getter = data_getter
        self.target_getter = target_getter
        self.stage_progress = stage_progress
        self.get_key_metric = get_key_metric
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}

    def fit(self, epochs):

        iterator = tqdm(range(epochs), dynamic_ncols=True)

        for epoch in iterator:
            output_train = train_epoch(
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                prefix = "[{}/{}]".format(epoch, epochs),
                stage_progress = self.stage_progress
            )

            output_test = test_epoch(
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                prefix = "[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                get_key_metric=self.get_key_metric
            )

            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train["loss"], output_test["metric"], output_test["loss"],
                    self.optimizer.param_groups[0]["lr"], epoch 
                )

            self.metrics["epoch"].append(epoch)
            self.metrics["train_loss"].append(output_train["loss"])
            self.metrics["test_loss"].append(output_test["loss"])
            self.metrics["test_metric"].append(output_test["metric"])

            self.lr_scheduler.step(output_train["loss"])

            progress_bar(iterator, epoch, output_train, output_test)

            if (epoch + 1) % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, self.model_name_prefix) + str(datetime.datetime.now())
                )

        return self.metrics