import torch
from torch.utils.data import Dataset

class PhoNERDataset(Dataset):
    def __init__(self, data, vocab, tag2id):
        self.data = data
        self.vocab = vocab
        self.tag2id = tag2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]

        input_ids = [
            self.vocab.get(w, self.vocab["<unk>"]) for w in words
        ]
        label_ids = [
            self.tag2id[t] for t in tags
        ]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(label_ids)
        }

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])

        input_ids.append(
            torch.cat([
                item["input_ids"],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )

        labels.append(
            torch.cat([
                item["labels"],
                torch.full((pad_len,), -100)
            ])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels)
    }
