import torch


def evaluate_loss(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            gt = torch.zeros(len(outputs), len(outputs[0])).to(device)
            for i, label in enumerate(labels):
                gt[i][label] = 1
            loss = loss_fn(outputs, gt)
            # loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
            #          + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())
            running_loss += loss.item()
    return running_loss / len(data_loader)
