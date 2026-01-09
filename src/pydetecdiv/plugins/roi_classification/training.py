import math
import sys
from typing import Callable, Any

import polars
import torch
import torchmetrics
from torch.amp import GradScaler, autocast

from pydetecdiv.plugins.roi_classification.evaluate import evaluate_metrics_seq2seq, evaluate_metrics_seq2one
from pydetecdiv.torch import ClassifierTrainingStats


def train_loop(training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader, model: torch.nn.Module,
               seq2one: bool, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, lambda1: float, lambda2: float,
               device: torch.device, train_stats: ClassifierTrainingStats) -> None:
    """
    Training loop wrapper function
    :param training_loader: the training data loader
    :param validation_loader: the validatino data loader
    :param model: the model
    :param seq2one: True if the classifier predicts a unique label from a sequence
    :param loss_fn: the loss function
    :param optimizer: the optimizer
    :param lambda1: the L1 regularization parameter
    :param lambda2: the L2 regularization parameter
    :param device: the device
    :param metric_fn: the metric used to evaluate the training performance
    :return: a polars DataFrame containing loss and metric values for training and validation sets
    """
    if seq2one:
        return train_loop_seq2one(training_loader, validation_loader, model, loss_fn, optimizer, lambda1, lambda2, device, train_stats)
    return train_loop_seq2seq(training_loader, validation_loader, model, loss_fn, optimizer, lambda1, lambda2, device, train_stats)


def train_loop_seq2one(training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
                       model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, lambda1: float,
                       lambda2: float,
                       device: torch.device, train_stats: ClassifierTrainingStats) -> None:
    """
    Training loop for seq to one models
    :param training_loader: the training data loader
    :param validation_loader: the validation data loader
    :param model: the model
    :param loss_fn: the loss function
    :param optimizer: the optimizer
    :param lambda1: the L1 regularization parameter
    :param lambda2: the L2 regularization parameter
    :param device: the device
    :param metric_fn: the metric used to evaluate the training performance
    :return: a polars DataFrame containing loss and metric values for training and validation sets
    """
    model.train()
    train_stats.metrics.reset()
    running_loss = 0.0
    scaler = GradScaler('cuda')

    for images, gt in training_loader:
        images, gt = images.to(device), gt.type(torch.LongTensor).to(device)
        # optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)

            if outputs.dim() == 2:
                loss = loss_fn(outputs, gt)
                train_stats.metrics.update(outputs, gt)
                # preds = outputs.argmax(dim=-1)
                # B, C = outputs.shape
            else:
                B, T, C = outputs.shape
                # preds = outputs[:, math.ceil(T / 2.0), :].argmax(dim=-1)
                loss = loss_fn(outputs[:, math.ceil(T / 2.0), :], gt)
                train_stats.metrics.update(outputs[:, math.ceil(T / 2.0), :], gt)

        loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                 + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    train_stats.log_metrics()
    train_stats.log_loss(avg_train_loss)

    avg_val_loss = evaluate_metrics_seq2one(model, validation_loader, loss_fn, lambda1, lambda2, device, train_stats.val_metrics)
    train_stats.log_val_metrics()
    train_stats.log_val_loss(avg_val_loss)


def train_loop_seq2seq(training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
                       model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, lambda1: float,
                       lambda2: float,
                       device: torch.device, train_stats: ClassifierTrainingStats) -> None:
    """
    Training loop for seq to seq models
    :param train_stats: Training statistics (history, metrics, etc.)
    :param training_loader: the training data loader
    :param validation_loader: the validation data loader
    :param model: the model
    :param loss_fn: the loss function
    :param optimizer: the optimizer
    :param lambda1: the L1 regularization parameter
    :param lambda2: the L2 regularization parameter
    :param device: the device
    :param metric_fn: the metric used to evaluate the training performance
    :return: a polars DataFrame containing loss and metric values for training and validation sets
    """
    model.train()
    train_stats.metrics.reset()
    running_loss = 0.0
    scaler = GradScaler('cuda')

    for images, gt in training_loader:
        images, gt = images.to(device), gt.type(torch.LongTensor).to(device)
        # optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(images)

            if outputs.dim() == 2:
                loss = loss_fn(outputs, gt)
                train_stats.metrics.update(outputs, gt)
                # B, C = outputs.shape
            else:
                B, T, C = outputs.shape
                loss = loss_fn(outputs.view(B * T, C), gt.view(B * T))
                train_stats.metrics.update(outputs.view(B * T, C), gt.view(B * T))
        # Apply L1 & L2 regularization
        loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                 + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    train_stats.log_metrics()
    train_stats.log_loss(avg_train_loss)

    # Validation phase
    avg_val_loss = evaluate_metrics_seq2seq(model, validation_loader, loss_fn, lambda1, lambda2, device, train_stats.val_metrics)
    train_stats.log_val_metrics()
    train_stats.log_val_loss(avg_val_loss)
