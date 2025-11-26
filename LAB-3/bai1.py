import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter

from data_utils.uit_vsfc import VSFCDataset, Vocab, collate_fn
from model.lstm import LstmMoodel
from model.logger_utils import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logger = get_logger(1)
logger.info("Start training bài 1 - LSTM(label - sentiment):")


# Evaluation function
def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for X, y, lengths in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(y.cpu().tolist())

    return {
        "f1": f1_score(true_labels, predictions, average="macro"),
        "precision": precision_score(true_labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(true_labels, predictions, average="macro", zero_division=0)
    }


# Load JSON helper
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

    # Create vocab và map nhãn
    vocab = Vocab(train_texts, train_labels)

    # Compute class weights
    counter = Counter(train_labels)
    total = sum(counter.values())

    weights = []
    for i in range(vocab.n_labels):
        # Nhãn i
        label = vocab.i2l[i]
        freq = counter[label]
        
        # Sqrt normalization + smoothing
        w = (total / freq)**0.5  # sqrt để hạn chế quá lớn
        weights.append(w)

    # Chuẩn hóa về [0, max_weight] để tránh số quá lớn
    max_w = max(weights)
    weights = [w / max_w for w in weights]

    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
   
    # Datasets
    train_ds = VSFCDataset(train_texts, train_labels, vocab)
    dev_ds = VSFCDataset(dev_texts, dev_labels, vocab)
    test_ds = VSFCDataset(test_texts, test_labels, vocab)

    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LstmMoodel(
        vocab_size=5000,
        hidden_size=256,
        n_layers=5,
        n_labels=vocab.n_labels
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

        # Evaluate on dev
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

    # Test set
    test_metrics = evaluate(test_dl, model)
