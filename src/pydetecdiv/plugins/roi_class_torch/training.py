import torch


def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device, lambda1, lambda2):
    model.train()
    running_loss = 0.0
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Apply L1 & L2 regularization
        loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                 + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())

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
            loss = loss_fn(outputs, labels)
            loss += (lambda1 * torch.abs(torch.cat([x.view(-1) for x in model.parameters()])).sum()
                     + lambda2 * torch.square(torch.cat([x.view(-1) for x in model.parameters()])).sum())
            val_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    avg_val_loss = val_loss / len(validation_loader)

    return {'train loss': avg_train_loss, 'val loss': avg_val_loss}
