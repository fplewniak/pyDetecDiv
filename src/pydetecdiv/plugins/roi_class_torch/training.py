import sys

import polars
import torch
from torch.amp import GradScaler, autocast

from pydetecdiv.plugins.roi_class_torch.evaluate import evaluate_metrics
from pydetecdiv.utils import flatten_list


# def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device, lambda1, lambda2):
def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0.0, 0.0
    scaler = GradScaler('cuda')

    for images, labels in training_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)

        with autocast('cuda'):
            outputs = model(images)
            gt = labels
            # print(f'{gt.shape=}')
            # print(f'{outputs.shape=}')
            # gt = torch.zeros(len(outputs), outputs.shape[-1]).to(device)
            # for i, label in enumerate(labels):
            #     gt[i][label] = 1
            # print(f'{outputs.shape=} {gt.shape=}', file=sys.stderr)
            if outputs.dim() == 2:
                loss = loss_fn(outputs, gt)
                B, C = outputs.shape
            else:
                # loss = loss_fn(outputs[:,int(outputs.shape[1]/2.0),...], gt)
                # loss = loss_fn(outputs[:, 0, ...], gt)
                B, T, C = outputs.shape
                loss = loss_fn(outputs.view(B * T, C), labels.view(B * T))
        # Apply L1 & L2 regularization
        # loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
        #          + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())
        #
        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item() * B

        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

        # for gt, p in zip(flatten_list(labels), flatten_list(preds)):
        #     print(gt.item(), p.item(), file=sys.stderr)
    # print(f'{outputs[0,0,...]=}')
    # print(f'{gt[0]=}')

    avg_train_loss = running_loss / len(training_loader)
    accuracy = correct / total

    # Validation phase
    avg_val_loss, val_accuracy = evaluate_metrics(model, validation_loader, loss_fn, device)

    return polars.DataFrame({'train loss': avg_train_loss, 'val loss': avg_val_loss,
                             'train accuracy': accuracy, 'val accuracy': val_accuracy,})

def train_testing_loop(training_loader, model, device):
    print('running testing train loop')
    print(f'{len(training_loader)=}')
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

    return {'train loss': 0.5, 'val loss': 0.5}
