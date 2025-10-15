import editdistance
import torch
from torchmetrics import Metric


class CERMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('errors', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_chars', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, predictions: list, targets: list):
        for pred, target in zip(predictions, targets):
            self.errors += editdistance.eval(pred[0], target)
            self.total_chars += len(target)

    def compute(self):
        return self.errors / self.total_chars

    def reset(self) -> None:
        self.errors.copy_(torch.tensor(0))
        self.total_chars.copy_(torch.tensor(0))
        super().reset()


if __name__ == '__main__':
    cer_metric = CERMetric()

    # Пример: предсказания и целевые значения
    preds = [['helloweensad adga'], ['wofdd']]
    targets = ['hallo', 'world']

    # Обновляем метрику
    cer_metric.update(preds, targets)

    # Вычисляем метрику
    result = cer_metric.compute()
    print(f"CER: {result.item()}")
