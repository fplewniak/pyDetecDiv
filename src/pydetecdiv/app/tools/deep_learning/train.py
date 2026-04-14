"""
Abstract ModelTrainer class
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import autocast, GradScaler
from pydetecdiv.app.tools.deep_learning.evaluate import ModelEvaluator

if TYPE_CHECKING:
    from pydetecdiv.app.tools.deep_learning import DeepTool


class ModelTrainer(ABC):
    """
    Abstract ModelTrainer class providing the basic functionalities for model training of deep learning tools
    """
    def __init__(self, tool: 'DeepTool'):
        self.tool = tool
        self.tool.command = 'train_model'

    @abstractmethod
    def train_model(self):
        """
        The global training procedure. This method should be implemented by subclasses to run the training loop for as many epochs
        as requested by the user.
        """

    @staticmethod
    def training_loop(training_dataloader, validation_dataloader, model, loss_fn, optimizer, device, train_stats):
        """
        The elementary training loop, run once per epoch on all batches
        """
        model.train()
        train_stats.metrics.reset()
        running_loss = 0.0
        scaler = GradScaler(device)

        for images, gt in training_dataloader:
            images, gt = images.to(device), gt.type(torch.LongTensor).to(device)
            # optimizer.zero_grad()

            with autocast(device.type):
                outputs = model(images)
                train_stats.metrics.update(outputs, gt)
                loss = loss_fn(outputs, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(training_dataloader)
        train_stats.log_metrics()
        train_stats.log_loss(avg_train_loss)

        avg_val_loss = ModelEvaluator.evaluate_model(validation_dataloader, model, loss_fn, device, train_stats.val_metrics)
        train_stats.log_val_metrics()
        train_stats.log_val_loss(avg_val_loss)
