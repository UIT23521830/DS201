import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from collections import Counter

from data_utils.logger import get_logger
from data_utils.PhoNER import PhoNERDataset, collate_fn
from model.transformer_ner import TransformerNER

# ======================
# CONFIG
# ======================
TRAIN_PATH = "dataset/PhoNER/train_word.json"
DEV_PATH   = "dataset/PhoNER/dev_word.json"
TEST_PATH  = "dataset/PhoNER/test_word.json"

BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(2)

# ======================
# LOAD DATA
# ======================
def load_phoner(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append((obj["words"], obj["tags"]))
    return data

def build_vocab_and_tags(data):
    wc = Counter()
    tag_set = set()

    for w, t in data:
        wc.update(w)
        tag_set.update(t)

    vocab = {"<pad>": 0, "<unk>": 1}
    for w in wc:
        vocab[w] = len(vocab)

    tag2id = {}
    for t in sorted(tag_set):
        tag2id[t] = len(tag2id)

    return vocab, tag2id

# ======================
# METRIC: F1
# ======================
def f1_score(preds, labels):
    tp, fp, fn = 0, 0, 0

    for p, l in zip(preds, labels):
        if l == -100:
            continue
        if p == l:
            tp += 1
        else:
            fp += 1
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return 2 * precision * recall / max(precision + recall, 1e-8)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            preds = logits.argmax(-1)

            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(y.view(-1).tolist())

    return f1_score(all_preds, all_labels)

# ======================
# MAIN
# ======================
def main():
    logger.info("Bài 2 - Transformer Encoder 3 layers cho PhoNER")

    train_data = load_phoner(TRAIN_PATH)
    dev_data   = load_phoner(DEV_PATH)
    test_data  = load_phoner(TEST_PATH)

    vocab, tag2id = build_vocab_and_tags(train_data)

    train_ds = PhoNERDataset(train_data, vocab, tag2id)
    dev_ds   = PhoNERDataset(dev_data, vocab, tag2id)
    test_ds  = PhoNERDataset(test_data, vocab, tag2id)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dl   = DataLoader(dev_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TransformerNER(
        vocab_size=len(vocab),
        num_tags=len(tag2id),
        n_layers=3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in train_dl:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_f1 = evaluate(model, dev_dl)
        logger.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {total_loss/len(train_dl):.4f} | "
            f"Dev F1: {dev_f1:.4f}"
        )

    test_f1 = evaluate(model, test_dl)
    logger.info(f"Final Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
