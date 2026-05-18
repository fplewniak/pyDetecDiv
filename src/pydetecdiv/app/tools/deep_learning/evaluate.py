"""
Abstract ModelEvaluator class
"""
from abc import ABC

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
    def evaluate_model(data_loader, model, loss_fn, device, metrics, regularization: int = 0, lambda_reg: float = 0.0):
        """
        Method to evaluate the model
        """
        model.eval()
        metrics.reset()
        running_loss = 0.0
        with torch.no_grad():
            for images, gt in data_loader:
                images, gt = images.to(device), gt.type(torch.LongTensor).to(device)
                with autocast(device.type):
                    outputs = model(images)
                    loss = loss_fn(outputs, gt)
                    metrics.update(outputs, gt)
                if regularization == 1:
                    loss += lambda_reg * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                elif regularization == 2:
                    loss += lambda_reg * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        del running_loss, loss, outputs, images, gt
        torch.cuda.empty_cache()

        return avg_loss
