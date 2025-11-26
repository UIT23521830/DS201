import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils.PhoNER import PhoNERDataset, collate_fn
from model.bilstm import BiLSTMEncoder
from model.logger_utils import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(3)
logger.info("Start training bài 3 (BiLSTMEncoder - PhoNER):")

# -------------------------------
# Load data helper
# -------------------------------
def load_data(max_len=100):
    paths = {
        "train": "dataset/PhoNER/train_syllable.json",
        "dev": "dataset/PhoNER/dev_syllable.json",
        "test": "dataset/PhoNER/test_syllable.json"
    }

    def _load_json(path):
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["words"][:max_len])
                labels.append(item["tags"][:max_len])
        return texts, labels

    train_texts, train_labels = _load_json(paths["train"])
    dev_texts, dev_labels = _load_json(paths["dev"])
    test_texts, test_labels = _load_json(paths["test"])

    # Build vocab
    vocab = {"PAD":0, "UNK":1}
    idx = 2
    for sentence in train_texts:
        for token in sentence:
            if token not in vocab:
                vocab[token] = idx
                idx += 1

    # Build label map
    label_set = set(l for labels in train_labels for l in labels)
    label_map = {l:i for i,l in enumerate(sorted(label_set))}

    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, vocab, label_map

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(dataloader, model):
    model.eval()
    true_labels_all, pred_labels_all = [], []

    with torch.no_grad():
        for X, y, lengths in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=-1)

            for i, l in enumerate(lengths):
                true_labels_all.extend(y[i, :l].cpu().tolist())
                pred_labels_all.extend(preds[i, :l].cpu().tolist())

    return {
        "f1": f1_score(true_labels_all, pred_labels_all, average="macro"),
        "precision": precision_score(true_labels_all, pred_labels_all, average="macro", zero_division=0),
        "recall": recall_score(true_labels_all, pred_labels_all, average="macro", zero_division=0)
    }

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, vocab, label_map = load_data(max_len=100)

    # Datasets & Dataloaders
    train_ds = PhoNERDataset(train_texts, train_labels, vocab, label_map, max_len=100)
    dev_ds = PhoNERDataset(dev_texts, dev_labels, vocab, label_map, max_len=100)
    test_ds = PhoNERDataset(test_texts, test_labels, vocab, label_map, max_len=100)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BiLSTMEncoder(
        vocab_size=len(vocab),
        embed_size=256,
        hidden_size=256,
        n_layers=5,
        n_labels=len(label_map)
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for X, y, lengths in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()

        metrics = evaluate(dev_dl, model)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

    # Test set
    test_metrics = evaluate(test_dl, model)