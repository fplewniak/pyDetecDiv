"""
Module for model evaluation
"""
import gc
import math
from typing import Any, Callable

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor
from torch.amp import autocast


def evaluate_metrics(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, seq2one: bool, loss_fn: torch.nn.Module,
                     lambda1: float, lambda2: float, device: torch.device,
                     metrics: torchmetrics.Metric) -> tuple[float, float]:
    """
    Wrapper function for evaluating metrics

    :param model: the model to evaluate metrics for
    :param data_loader: the data loader to evaluate metrics with
    :param seq2one: if the model is a seq 2 one
    :param loss_fn: the loss function (nn.Module)
    :param lambda1: the L1 regularization factor
    :param lambda2: the L2 regularization factor
    :param device: the device
    :param metrics: the metrics to evaluate
    :return: the average loss and requested metrics
    """
    if seq2one:
        return evaluate_metrics_seq2one(model, data_loader, loss_fn, lambda1, lambda2, device, metrics)
    return evaluate_metrics_seq2seq(model, data_loader, loss_fn, lambda1, lambda2, device, metrics)


def evaluate_metrics_seq2one(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
                             lambda1: float, lambda2: float, device: torch.device,
                             metrics: torchmetrics.Metric) -> tuple[float, float]:
    """
    Evaluating metrics for a seq to one classifier

    :param model: the model to evaluate metrics for
    :param data_loader: the data loader to evaluate metrics with
    :param loss_fn: the loss function (nn.Module)
    :param lambda1: the L1 regularization factor
    :param lambda2: the L2 regularization factor
    :param device: the device
    :param metrics: the metrics to evaluate
    :return: the average loss and requested metrics
    """
    model.eval()
    metrics.reset()
    running_loss = 0.0
    # scaler = GradScaler('cuda')
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            with autocast('cuda'):
                outputs = model(images)
                gt = labels

                if outputs.dim() == 2:
                    loss = loss_fn(outputs, gt)
                    metrics.update(outputs, gt)
                    # preds = outputs.argmax(dim=-1)
                    # B, C = outputs.shape
                else:
                    _, T, _ = outputs.shape
                    # preds = outputs[:, math.ceil(T / 2.0), :].argmax(dim=-1)
                    loss = loss_fn(outputs[:, math.ceil(T / 2.0), :], gt)
                    metrics.update(outputs[:, math.ceil(T / 2.0), :], gt)

            loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                     + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

            # running_loss += loss.item() * B
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        avg_metric = metrics.compute()

        return avg_loss, avg_metric


def evaluate_metrics_seq2seq(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
                             lambda1: float, lambda2: float, device: torch.device,
                             metrics: torchmetrics.Metric) -> tuple[float, float]:
    """
    Evaluating metrics for a seq to seq classifier

    :param model: the model to evaluate metrics for
    :param data_loader: the data loader to evaluate metrics with
    :param loss_fn: the loss function (nn.Module)
    :param lambda1: the L1 regularization factor
    :param lambda2: the L2 regularization factor
    :param device: the device
    :param metrics: the metrics to evaluate
    :return: the average loss and requested metrics
    """
    model.eval()
    metrics.reset()
    running_loss = 0.0
    # scaler = GradScaler('cuda')
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            with autocast('cuda'):
                outputs = model(images)
                if outputs.dim() == 2:
                    loss = loss_fn(outputs, labels)
                    metrics.update(outputs, labels)
                    B, C = outputs.shape
                else:
                    B, T, C = outputs.shape
                    loss = loss_fn(outputs.view(B * T, C), labels.view(B * T))
                    metrics.update(outputs.view(B * T, C), labels.view(B * T))
            loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                     + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

            # running_loss += loss.item() * B
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        avg_metric = metrics.compute()

        return avg_loss, avg_metric


def get_pred_gt(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, seq2one: bool,
                device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Wrapper function returning prediction and corresponding ground truth

    :param model: the model to evaluate metrics for
    :param data_loader: the data loader to evaluate metrics with
    :param seq2one: if the model is a seq 2 one
    :param device: the device
    :return: the predictions and ground truth
    """
    if seq2one:
        return get_pred_gt_seq2one(model, data_loader, device)
    return get_pred_gt_seq2seq(model, data_loader, device)


def get_pred_gt_seq2one(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                        device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Function returning the predictions and ground truth for a seq to one model

    :param model: the model to evaluate metrics for
    :param data_loader: the data loader to evaluate metrics with
    :param device: the device
    :return: the predictions and ground truth
    """
    model.eval()
    predictions: Tensor | None = None
    ground_truth: Tensor | None = None

    for img, targets in data_loader:
        with autocast('cuda'):
            img: Tensor = img.to(device)
            ground_truth: Tensor = targets.type(torch.LongTensor).to(device)
            outputs = model(img)
        if outputs.dim() == 2:
            predictions = outputs.argmax(dim=-1)
        else:
            _, T, _ = outputs.shape
            predictions = outputs[:, math.ceil(T / 2), :].argmax(dim=-1)

    return predictions, ground_truth


def get_pred_gt_seq2seq(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                        device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Function returning the predictions and ground truth for a seq to seq model

    :param model: the model to evaluate metrics for
    :param data_loader: the data loader to evaluate metrics with
    :param device: the device
    :return: the predictions and ground truth
    """
    model.eval()

    predictions:Tensor | None  = None
    ground_truth: Tensor | None = None

    for img, targets in data_loader:
        with autocast('cuda'):
            img: Tensor = img.to(device)
            targets: Tensor = targets.type(torch.LongTensor).to(device)
            outputs = model(img)
        if outputs.dim() == 2:
            B, C = outputs.shape
            T = 1
        else:
            B, T, C = outputs.shape

        if predictions is None:
            predictions = outputs.view(B * T, C).argmax(axis=-1)
            ground_truth = targets.view(B * T)
        else:
            predictions = torch.cat((predictions, outputs.view(B * T, C).argmax(axis=-1)))
            ground_truth = torch.cat((ground_truth, targets.view(B * T)))

    return predictions, ground_truth


def evaluate_model(model: torch.nn.Module, class_names: list[str], data_loader: torch.utils.data.DataLoader, seqlen: int,
                   seq2one: bool, device: torch.device) -> tuple[dict[str, list[str | int]], Any, Any]:
    """
    Function to evaluate model

    :param model: the model to evaluate in its last state
    :param class_names: the list of class names
    :param data_loader: the data loader providing the data to evaluate the model on
    :param seqlen: the sequence length
    :param seq2one: if True, the model is seq to one, it is seq to seq otherwise
    :param device: the device
    :return: the statistics, predictions and ground truth for the last model and the best checkpoint
    """
    predictions, ground_truth = get_pred_gt(model, data_loader, seq2one, device)
    labels = list(range(len(class_names)))

    precision, recall, fscore, support = precision_recall_fscore_support(ground_truth.cpu(), predictions.cpu(), labels=labels,
                                                                         zero_division=np.nan)

    col_names = ['stats'] + class_names
    p = ['precision'] + precision.tolist()
    r = ['recall'] + recall.tolist()
    f = ['fscore'] + fscore.tolist()
    s = ['support'] + [int(s) for s in support]

    stats = {col_name: [p[i], r[i], f[i], s[i]] for i, col_name in enumerate(col_names)}

    return stats, ground_truth, predictions
