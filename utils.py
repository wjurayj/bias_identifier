import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class TokenizedScalingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[True, False])
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
        'confusion': cm
    }
