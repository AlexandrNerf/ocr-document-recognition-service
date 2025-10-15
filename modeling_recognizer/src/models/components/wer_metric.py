import torch
from torchmetrics import Metric


class WERMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('errors', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: list, targets: list):
        for pred, target in zip(preds, targets):
            self.errors += pred[0].strip() != target.strip()
            self.total_samples += 1

    def compute(self):
        return self.errors.float() / self.total_samples

    def reset(self) -> None:
        self.errors.copy_(torch.tensor(0))
        self.total_samples.copy_(torch.tensor(0))
        super().reset()
