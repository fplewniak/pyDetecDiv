import torch
import torch.nn.functional as F

class Weighed_Accuracy:
    def __init__(self, weighted_mean=True):
        self.weighted_mean = weighted_mean
        self.reset_sampling()

    def sampling(self, outputs, targets):
        self.sample_outputs = torch.cat([self.sample_outputs, outputs.detach()])
        self.sample_targets = torch.cat([self.sample_targets, targets])

    def reset_sampling(self):
        self.sample_outputs = torch.tensor([])
        self.sample_targets = torch.tensor([])

    def get_accuracy(self):
        if self.sample_outputs.dim() == 1:
            return None
        outputs = self.sample_outputs
        targets = self.sample_targets
        N, C = outputs.size(0), outputs.size(1)
        outputs = outputs.view(N, C, -1)
        targets = targets.view(N, -1)
        pred = torch.argmax(outputs, dim=1)
        correct = (pred == targets).unsqueeze(1)
        target_onehot = F.one_hot(targets.to(torch.int64), num_classes=C).transpose(1, 2)
        correct = correct.expand_as(target_onehot) * target_onehot
        total = target_onehot.sum((0, 2))
        correct = correct.sum((0, 2)) / total
        if self.weighted_mean:
            return correct.mean()
        else:
            return correct
