import json
from torch.utils.data import Dataset
import torch
from collections import Counter

class PhoNERDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_map, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx][:self.max_len]
        labels = self.labels[idx][:self.max_len]

        token_ids = [self.vocab.get(t, self.vocab["UNK"]) for t in tokens]
        label_ids = [self.label_map[l] for l in labels]

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

def collate_fn(batch):
    token_ids, label_ids = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in token_ids])
    token_ids_padded = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)
    label_ids_padded = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
    return token_ids_padded, label_ids_padded, lengths

def load_data(max_len=100):
    """
    Load PhoNER train/dev/test datasets, tạo vocab và label_map
    """
    def read_json_lines(path):
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj["words"])
                labels.append(obj["tags"])
        return texts, labels

    train_texts, train_labels = read_json_lines("dataset/PhoNER/train_syllable.json")
    dev_texts, dev_labels = read_json_lines("dataset/PhoNER/dev_syllable.json")
    test_texts, test_labels = read_json_lines("dataset/PhoNER/test_syllable.json")

    # Build vocab (token -> idx)
    tokens = [t for sent in train_texts for t in sent]
    vocab = {t:i+1 for i, t in enumerate(sorted(set(tokens)))}  # +1 để 0 là padding
    vocab["UNK"] = len(vocab) + 1

    # Build label map (label -> idx)
    labels = [l for sent in train_labels for l in sent]
    label_set = sorted(set(labels))
    label_map = {l:i for i,l in enumerate(label_set)}

    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, vocab, label_map
