"""
Abstract ModelEvaluator class
"""
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

import torch
from torch import autocast

if TYPE_CHECKING:
    from pydetecdiv.app.tools.deep_learning import DeepTool


class ModelEvaluator(ABC):
    """
    Abstract ModelEvaluator class providing the basic functionalities for deep learning model evaluation
    """
    def __init__(self, tool: 'DeepTool'):
        self.tool = tool
        self.tool.command = 'evaluate_model'

    @staticmethod
    def evaluate_model(data_loader, model, loss_fn, device, metrics):
        """
        Method to evaluate the model
        """
        model.eval()
        metrics.reset()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
                with autocast(device.type):
                    outputs = model(images)
                    gt = labels - 1
                    loss = loss_fn(outputs, gt)
                    metrics.update(outputs, gt)

                running_loss += loss.item()

        return running_loss / len(data_loader)
