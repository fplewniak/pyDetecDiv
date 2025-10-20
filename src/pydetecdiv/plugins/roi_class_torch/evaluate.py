import gc
import math
from datetime import datetime

import numpy as np
import polars
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.amp import autocast
from torch.amp import GradScaler

from pydetecdiv.utils import flatten_list

def evaluate_metrics(model, data_loader, seq2one, loss_fn, lambda1, lambda2, device):
    if seq2one:
        return evaluate_metrics_seq2one(model, data_loader, loss_fn, lambda1, lambda2, device)
    return evaluate_metrics_seq2seq(model, data_loader, loss_fn, lambda1, lambda2, device)

def evaluate_metrics_seq2one(model, data_loader, loss_fn, lambda1, lambda2, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0.0, 0.0
    # scaler = GradScaler('cuda')
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
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

        # running_loss += loss.item() * B
        running_loss += loss.item()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        return running_loss / len(data_loader), correct / total


def evaluate_metrics_seq2seq(model, data_loader, loss_fn, lambda1, lambda2, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0.0, 0.0
    # scaler = GradScaler('cuda')
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            with autocast('cuda'):
                outputs = model(images)
                if outputs.dim() == 2:
                    loss = loss_fn(outputs, labels)
                    B, C = outputs.shape
                else:
                    B, T, C = outputs.shape
                    loss = loss_fn(outputs.view(B * T, C), labels.view(B * T))
            loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                     + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

            # running_loss += loss.item() * B
            running_loss += loss.item()

            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return running_loss / len(data_loader), correct / total

def get_pred_gt(model, data_loader, seq2one, device):
    if seq2one:
        return get_pred_gt_seq2one(model, data_loader, device)
    return get_pred_gt_seq2seq(model, data_loader, device)


def get_pred_gt_seq2one(model, data_loader, device):
    model.eval()
    predictions, ground_truth = None, None
    for img, targets in data_loader:
        with autocast('cuda'):
            img, ground_truth = img.to(device), targets.type(torch.LongTensor).to(device)
            outputs = model(img)
        if outputs.dim() == 2:
            predictions = outputs.argmax(dim=-1)
        else:
            B, T, C = outputs.shape
            predictions = outputs[:, math.ceil(T / 2), :].argmax(dim=-1)

    return predictions, ground_truth


def get_pred_gt_seq2seq(model, data_loader, device):
    model.eval()

    predictions, ground_truth = None, None
    for img, targets in data_loader:
        with autocast('cuda'):
            img, targets = img.to(device), targets.type(torch.LongTensor).to(device)
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

def evaluate_model(model, checkpoint_filepath, class_names, data_loader, seqlen, seq2one, device):
    predictions, ground_truth = get_pred_gt(model, data_loader, seq2one, device)

    model = torch.jit.load(checkpoint_filepath)
    best_predictions, best_gt = get_pred_gt(model, data_loader,  seq2one, device)

    del(model)
    gc.collect()

    labels = list(range(len(class_names)))

    precision, recall, fscore, support = precision_recall_fscore_support(ground_truth.cpu(), predictions.cpu(), labels=labels,
                                                                         zero_division=np.nan)

    best_precision, best_recall, best_fscore, best_support = precision_recall_fscore_support(best_gt.cpu(),
                                                                                             best_predictions.cpu(),
                                                                                             labels=labels,
                                                                                             zero_division=np.nan)

    col_names = ['stats'] + class_names
    p = ['precision'] + precision.tolist()
    r = ['recall'] + recall.tolist()
    f = ['fscore'] + fscore.tolist()
    s = ['support'] + [int(s) for s in support]
    bp = ['precision'] + best_precision.tolist()
    br = ['recall'] + best_recall.tolist()
    bf = ['fscore'] + best_fscore.tolist()
    bs = ['support'] + [int(s) for s in best_support]

    stats = {'last_stats': {col_name: [p[i], r[i], f[i], s[i]] for i, col_name in enumerate(col_names)},
             'best_stats': {col_name: [bp[i], br[i], bf[i], bs[i]] for i, col_name in enumerate(col_names)}
             }

    return stats, ground_truth, predictions, best_gt, best_predictions
