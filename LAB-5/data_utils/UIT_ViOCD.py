import json
import torch

class UITViOCDDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, vocab, domain2id=None):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = list(raw.values())
        self.vocab = vocab

        if domain2id is None:
            domains = sorted({item["domain"] for item in self.samples})
            self.domain2id = {d: i for i, d in enumerate(domains)}
        else:
            self.domain2id = domain2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": self.vocab.encode(item["review"]),
            "label": self.domain2id[item["domain"]]
        }


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []

    for b in batch:
        ids = b["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [0] * pad_len)
        labels.append(b["label"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }
