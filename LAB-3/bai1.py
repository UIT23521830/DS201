import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils.PhoNER import PhoNERDataset, collate_fn, load_data
from model.bilstm import BiLSTMEncoder
from model.logger_utils import get_logger

# -------------------------------
# Device & cuDNN benchmark
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------------------
# Logger
# -------------------------------
logger = get_logger(3)
logger.info("Start training Bài 3 (NER - BiLSTMEncoder 5 lớp)")

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
    # Load data
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, vocab, label_map = load_data()

    # Datasets & DataLoaders với max_len=100, batch_size=64
    max_len = 100
    batch_size = 64

    train_ds = PhoNERDataset(train_texts, train_labels, vocab, label_map, max_len=max_len)
    dev_ds = PhoNERDataset(dev_texts, dev_labels, vocab, label_map, max_len=max_len)
    test_ds = PhoNERDataset(test_texts, test_labels, vocab, label_map, max_len=max_len)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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

    # Final Test
    test_metrics = evaluate(test_dl, model)

