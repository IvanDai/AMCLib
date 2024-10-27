import torch
import torch.nn as nn

# == Metrics Defination ==
class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.correct = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0),requires_grad=False)
    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0] 
        self.correct += m 
        self.total += n
        return m/n
    def compute(self):
        return self.correct.float() / self.total 
    def reset(self):
        self.correct -= self.correct
        self.total -= self.total