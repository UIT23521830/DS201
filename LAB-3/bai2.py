import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils.uit_vsfc import VSFCDataset, Vocab, collate_fn
from model.gru import GRUModel
from model.logger_utils import get_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(2)
logger.info("Start training bài 2 - GRU(label - sentiment):")



# Evaluation function

def evaluate(dataloader, model):
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for X, y, lengths in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=-1)

            preds.extend(pred.cpu().tolist())
            gold.extend(y.cpu().tolist())

    return {
        "f1": f1_score(gold, preds, average="macro"),
        "precision": precision_score(gold, preds, average="macro"),
        "recall": recall_score(gold, preds, average="macro")
    }



# Load JSON

def load_json(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        texts.append(item["sentence"])
        labels.append(item["sentiment"])

    return texts, labels


# Main
if __name__ == "__main__":

    # Load data
    train_texts, train_labels = load_json("dataset/UIT-VSFC/UIT-VSFC-train.json")
    dev_texts, dev_labels = load_json("dataset/UIT-VSFC/UIT-VSFC-dev.json")
    test_texts, test_labels = load_json("dataset/UIT-VSFC/UIT-VSFC-test.json")

    # Build vocab
    vocab = Vocab(train_texts, train_labels)

    # Dataset
    train_ds = VSFCDataset(train_texts, train_labels, vocab)
    dev_ds = VSFCDataset(dev_texts, dev_labels, vocab)
    test_ds = VSFCDataset(test_texts, test_labels, vocab)

    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Model
    model = GRUModel(
        vocab_size=5000,        # encode_sentence() giữ nguyên 5000
        hidden_size=256,
        n_layers=5,
        n_labels=vocab.n_labels
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10


    # Training loop
    for epoch in range(num_epochs):
        model.train()

        for X, y, lengths in train_dl:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        metrics = evaluate(dev_dl, model)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

        # print(
        #     f"Epoch {epoch+1}/{num_epochs} | "
        #     f"F1: {metrics['f1']:.4f} | "
        #     f"Precision: {metrics['precision']:.4f} | "
        #     f"Recall: {metrics['recall']:.4f}"
        # )

    test_metrics = evaluate(test_dl, model)

