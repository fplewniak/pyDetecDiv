from datetime import datetime

import numpy as np
import polars
import torch
from sklearn.metrics import precision_recall_fscore_support


def evaluate_loss(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            outputs = model(images)
            gt = labels
            # gt = torch.zeros(len(outputs), outputs.shape[-1]).to(device)
            # for i, label in enumerate(labels):
            #     gt[i][label] = 1
            if outputs.dim() == 2:
                loss = loss_fn(outputs, gt)
            else:
                loss = loss_fn(outputs[:, 0, ...], gt)
            # loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
            #          + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())
            running_loss += loss.item()
        # print(f'{outputs[0,0,...]=}')
        # print(f'{gt[0]=}')
    return running_loss / len(data_loader)


def evaluate_model(model, checkpoint_filepath, class_names, data_loader, seqlen, device):
    ground_truth = [label for batch in [y for x, y in data_loader] for label in batch]
    model.eval()
    if seqlen == 0:
        print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction for last model')
        predictions = [label.cpu() for batch in [model(img.to(device)).argmax(axis=-1) for img, target in data_loader] for label
                       in batch]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction for best model')
        model = torch.jit.load(checkpoint_filepath)
        best_predictions = [label.cpu() for batch in [model(img.to(device)).argmax(axis=-1) for img, target in data_loader] for
                            label in batch]
    else:
        print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction for last model')
        predictions = [label.cpu()[0] for batch in [model(img.to(device)).argmax(axis=-1) for img, target in data_loader] for
                       label in batch]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction for best model')
        model = torch.jit.load(checkpoint_filepath)
        best_predictions = [label.cpu()[0] for batch in [model(img.to(device)).argmax(axis=-1) for img, target in data_loader]
                            for label in batch]
        # ground_truth = [label for seq in ground_truth for label in seq]

    labels = list(range(len(class_names)))
    precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, predictions, labels=labels,
                                                                         zero_division=np.nan)
    best_precision, best_recall, best_fscore, best_support = precision_recall_fscore_support(ground_truth, best_predictions,
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

    return stats, ground_truth, predictions, best_predictions
