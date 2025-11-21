import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        # input: (batch_size * seq_len, num_classes)
        # target: (batch_size * seq_len)
        probs = F.softmax(input, dim=1)
        log_probs = torch.log(probs)
        probs_t = probs.gather(1, target.unsqueeze(1))  # (batch_size * seq_len, 1)

        # Compute focal loss
        focal_weight = (1 - probs_t) ** self.gamma
        loss = -focal_weight * log_probs.gather(1, target.unsqueeze(1))

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            alpha_t = self.alpha.gather(0, target.data.view(-1))
            loss = loss * alpha_t

        # Reduce loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
