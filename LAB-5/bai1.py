# bai1.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from sklearn.metrics import f1_score

from data_utils.logger import get_logger
from data_utils.vocab import Vocab
from data_utils.UIT_ViOCD import UITViOCDDataset, collate_fn
from model.classification import TransformerClassifier

# ======================
# CONFIG
# ======================
TRAIN_JSON = "dataset/UIT_ViOCD/train.json"
DEV_JSON   = "dataset/UIT_ViOCD/dev.json"
TEST_JSON  = "dataset/UIT_ViOCD/test.json"

BATCH_SIZE = 16
D_MODEL = 256
N_ENCODER = 3
LR = 1e-4
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logger → lưu bai1.log
logger = get_logger("bai1")

# ======================
# LOAD RAW TEXT (BUILD VOCAB)
# ======================
def load_reviews(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["review"] for item in data.values()]

# ======================
# EVALUATION (F1 MACRO)
# ======================
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro")
    return f1

# ======================
# MAIN
# ======================
def main():
    logger.info("Start Bài 1 - Transformer Encoder Domain Classification (UIT-ViOCD)")

    # ---- build vocab
    logger.info("Loading training text for vocab...")
    train_texts = load_reviews(TRAIN_JSON)

    logger.info("Building vocabulary...")
    vocab = Vocab(train_texts)
    logger.info(f"Vocab size = {len(vocab)}")

    # ---- datasets (share domain2id)
    train_ds = UITViOCDDataset(TRAIN_JSON, vocab)
    domain2id = train_ds.domain2id

    dev_ds  = UITViOCDDataset(DEV_JSON, vocab, domain2id)
    test_ds = UITViOCDDataset(TEST_JSON, vocab, domain2id)

    logger.info(f"Number of domains = {len(domain2id)}")

    # ---- dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_dl = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- model
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(domain2id)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_dev_f1 = 0.0

    # ======================
    # TRAINING LOOP
    # ======================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        dev_f1 = evaluate(model, dev_dl)

        logger.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Dev F1: {dev_f1:.4f}"
        )

        # ---- save best model
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), "best_bai1_model.pt")

    # ======================
    # FINAL TEST
    # ======================
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load("best_bai1_model.pt", map_location=device))
    test_f1 = evaluate(model, test_dl)

    logger.info(f"Final Test F1 (macro): {test_f1:.4f}")

# ======================
if __name__ == "__main__":
    main()
