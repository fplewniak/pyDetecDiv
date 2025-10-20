import math
import sys

import polars
import torch
from torch import unsqueeze
from torch.amp import GradScaler, autocast

from pydetecdiv.plugins.roi_class_torch.evaluate import evaluate_metrics_seq2seq, evaluate_metrics_seq2one
from pydetecdiv.utils import flatten_list

def train_loop(training_loader, validation_loader, model, seq2one, loss_fn, optimizer, lambda1, lambda2, device):
    if seq2one:
        return train_loop_seq2one(training_loader, validation_loader, model, loss_fn, optimizer, lambda1, lambda2, device)
    return train_loop_seq2seq(training_loader, validation_loader, model, loss_fn, optimizer, lambda1, lambda2, device)

def train_loop_seq2one(training_loader, validation_loader, model, loss_fn, optimizer, lambda1, lambda2, device):
    model.train()
    running_loss = 0.0
    correct, total = 0.0, 0.0
    # scaler = GradScaler('cuda')

    for images, labels in training_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)
            gt = labels

            if outputs.dim() == 2:
                loss = loss_fn(outputs, gt)
                preds = outputs.argmax(dim=-1)
                B, C = outputs.shape
            else:
                B, T, C = outputs.shape
                preds = outputs[:, math.ceil(T / 2.0), :].argmax(dim=-1)
                loss = loss_fn(outputs[:, math.ceil(T / 2.0), :], gt)

        loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                 + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # running_loss += loss.item() * B
        running_loss += loss.item()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_train_loss = running_loss / total
        accuracy = correct / total

        avg_val_loss, val_accuracy = evaluate_metrics_seq2one(model, validation_loader, loss_fn, lambda1, lambda2, device)

        return polars.DataFrame({'train loss': avg_train_loss, 'val loss': avg_val_loss,
                             'train accuracy': accuracy, 'val accuracy': val_accuracy,})


def train_loop_seq2seq(training_loader, validation_loader, model, loss_fn, optimizer, lambda1, lambda2, device):
    model.train()
    running_loss = 0.0
    correct, total = 0.0, 0.0
    # scaler = GradScaler('cuda')

    for images, labels in training_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(images)
            gt = labels

            if outputs.dim() == 2:
                loss = loss_fn(outputs, gt)
                B, C = outputs.shape
            else:
                B, T, C = outputs.shape
                loss = loss_fn(outputs.view(B * T, C), labels.view(B * T))
        # Apply L1 & L2 regularization
        loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                 + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # running_loss += loss.item() * B
        running_loss += loss.item()

        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_train_loss = running_loss / len(training_loader)
    accuracy = correct / total

    # Validation phase
    avg_val_loss, val_accuracy = evaluate_metrics_seq2seq(model, validation_loader, loss_fn, lambda1, lambda2, device)

    return polars.DataFrame({'train loss': avg_train_loss, 'val loss': avg_val_loss,
                             'train accuracy': accuracy, 'val accuracy': val_accuracy,})

def train_testing_loop(training_loader, model, device):
    print('running testing train loop')
    print(f'{len(training_loader)=}')
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

    return {'train loss': 0.5, 'val loss': 0.5}
