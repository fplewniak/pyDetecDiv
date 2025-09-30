import polars
import torch


# def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device, lambda1, lambda2):
def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        gt = torch.zeros(len(outputs), len(outputs[0])).to(device)
        for i, label in enumerate(labels):
            gt[i][label] = 1
        loss = loss_fn(outputs, gt)
        # Apply L1 & L2 regularization
        # loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
        #          + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            gt = torch.zeros(len(outputs), len(outputs[0])).to(device)
            for i, label in enumerate(labels):
                gt[i][label] = 1
            loss = loss_fn(outputs, gt)
            # loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
            #          + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())
            val_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    avg_val_loss = val_loss / len(validation_loader)

    return polars.DataFrame({'train loss': avg_train_loss, 'val loss': avg_val_loss})

def train_testing_loop(training_loader, model, device):
    print('running testing train loop')
    print(f'{len(training_loader)=}')
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

    return {'train loss': 0.5, 'val loss': 0.5}
