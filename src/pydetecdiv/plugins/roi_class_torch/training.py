import sys

import polars
import torch

from pydetecdiv.plugins.roi_class_torch.evaluate import evaluate_loss


# def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device, lambda1, lambda2):
def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in training_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        outputs = model(images)
        gt = labels
        # gt = torch.zeros(len(outputs), outputs.shape[-1]).to(device)
        # for i, label in enumerate(labels):
        #     gt[i][label] = 1
        # print(f'{outputs.shape=} {gt.shape=}', file=sys.stderr)
        if outputs.dim() == 2:
            loss = loss_fn(outputs, gt)
        else:
            loss = loss_fn(outputs[:,0,...], gt)
        # Apply L1 & L2 regularization
        # loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
        #          + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    # print(f'{outputs[0,0,...]=}')
    # print(f'{gt[0]=}')

    avg_train_loss = running_loss / len(training_loader)
    # Validation phase
    avg_val_loss = evaluate_loss(model, validation_loader, loss_fn, device)

    return polars.DataFrame({'train loss': avg_train_loss, 'val loss': avg_val_loss})

def train_testing_loop(training_loader, model, device):
    print('running testing train loop')
    print(f'{len(training_loader)=}')
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

    return {'train loss': 0.5, 'val loss': 0.5}
