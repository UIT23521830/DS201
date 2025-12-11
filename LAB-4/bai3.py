# bai3.py
import os
import json
import torch
from torch.utils.data import DataLoader
from torch import optim
from rouge_score import rouge_scorer

from data_utils.logger import get_logger
from data_utils.vocab import Vocab
from data_utils.PhoMT import PhoMTDataset, collate_fn
from model.LSTM_Luong_attn import Seq2seqLSTMLuong  # NEW model for Bài 3

# config
TRAIN_JSON = "dataset/small-train.json"
DEV_JSON = "dataset/small-dev.json"
TEST_JSON = "dataset/small-test.json"

BATCH_SIZE = 8
D_MODEL = 256
N_ENCODER = 3
N_DECODER = 3
LR = 1e-3
EPOCHS = 3
CACHE_VOCAB = "vocab_cache.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(3)

def load_json_pair(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    src = [item["english"] for item in data]
    tgt = [item["vietnamese"] for item in data]
    return src, tgt

def evaluate(model, dataloader, vocab):
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    total = 0.0
    n = 0
    with torch.no_grad():
        for src, tgt, _, _ in dataloader:
            src = src.to(device)
            preds = model.predict(src)
            for p_ids, g_ids in zip(preds, tgt):
                pred = vocab.decode([int(x) for x in p_ids.tolist()], language="tgt")
                gold = vocab.decode([int(x) for x in g_ids.tolist()], language="tgt")
                score = scorer.score(gold, pred)["rougeL"].fmeasure
                total += score
                n += 1
    return total / max(n, 1)

def main():
    logger.info("Start training Bài 3 - Seq2Seq LSTM (Luong Attention) on PhoMT")

    train_src, train_tgt = load_json_pair(TRAIN_JSON)
    dev_src, dev_tgt = load_json_pair(DEV_JSON)
    test_src, test_tgt = load_json_pair(TEST_JSON)

    if os.path.exists(CACHE_VOCAB):
        logger.info("Loading vocab from cache...")
        vocab = Vocab(train_src, train_tgt, cache_path=CACHE_VOCAB)
    else:
        logger.info("Building vocab from train set...")
        vocab = Vocab(train_src, train_tgt, cache_path=CACHE_VOCAB)
        logger.info(f"vocab sizes src={vocab.total_src_tokens} tgt={vocab.total_tgt_tokens}")

    train_ds = PhoMTDataset(TRAIN_JSON, vocab)
    dev_ds = PhoMTDataset(DEV_JSON, vocab)
    test_ds = PhoMTDataset(TEST_JSON, vocab)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Seq2seqLSTMLuong(d_model=D_MODEL, vocab=vocab, n_encoder=N_ENCODER, n_decoder=N_DECODER, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for i, (src, tgt, _, _) in enumerate(train_dl, start=1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            # teacher forcing: input tgt without last token
            loss = model(src, tgt[:, :-1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 0:
                logger.info(f"Epoch {epoch} step {i}/{len(train_dl)} loss {running_loss / i:.4f}")

        avg_loss = running_loss / len(train_dl)
        rouge_l = evaluate(model, dev_dl, vocab)
        logger.info(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | ROUGE-L dev: {rouge_l:.4f}")

    test_score = evaluate(model, test_dl, vocab)
    logger.info(f"Final Test ROUGE-L: {test_score:.4f}")

if __name__ == "__main__":
    main()
