# Accuracy and macro precision, recall, and F1.
# Each species counts equally in the macro averages.
import torch


def classification_metrics(predictions, targets, num_classes):
    predictions = torch.as_tensor(predictions, dtype=torch.long).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    matrix = torch.bincount(
        targets * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    true_positive = matrix.diag().float()
    predicted = matrix.sum(dim=0).float()
    support = matrix.sum(dim=1).float()
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    precision[predicted > 0] = true_positive[predicted > 0] / predicted[predicted > 0]
    recall[support > 0] = true_positive[support > 0] / support[support > 0]
    f1 = torch.zeros(num_classes)
    denominator = precision + recall
    f1[denominator > 0] = 2 * precision[denominator > 0] * recall[denominator > 0] / denominator[denominator > 0]

    total = matrix.sum().item()
    return {
        "accuracy": true_positive.sum().item() / total,
        "precision_macro": precision.mean().item(),
        "recall_macro": recall.mean().item(),
        "f1_macro": f1.mean().item(),
        "confusion_matrix": matrix.tolist(),
    }
