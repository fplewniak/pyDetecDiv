"""
Extra loss functions that are not provided by the Pytorch package
"""
import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Focal loss function, extending Cross entropy loss, and gives more emphasis to classes that are poorly predicted, i.e. whose
    correct label is predicted with a small probability. The gamma parameter defines the strength of this emphasis. If gamma = 0,
    this function is equivalent to cross entropy. The alpha parameter defines class weights to address unbalanced classes issues.
    """
    def __init__(self, alpha: list[float] = None, gamma: float = 2, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (batch_size * seq_len, num_classes)
        # target: (batch_size * seq_len)
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs)
        probs_t = probs.gather(1, target.unsqueeze(1))  # (batch_size * seq_len, 1)

        # Compute focal loss
        focal_weight = (1 - probs_t) ** self.gamma
        loss = -focal_weight * log_probs.gather(1, target.unsqueeze(1))

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            alpha_t = self.alpha.gather(0, target.data.view(-1))
            loss = loss * alpha_t

        # Reduce loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
