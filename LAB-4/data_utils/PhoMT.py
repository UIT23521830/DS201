# data_utils/PhoMT.py
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PhoMTDataset(Dataset):
    def __init__(self, json_path, vocab):
        self.vocab = vocab
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.src_texts = [item["english"] for item in data]
        self.tgt_texts = [item["vietnamese"] for item in data]

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_ids = self.vocab.encode(self.src_texts[idx], language="src")
        tgt_ids = self.vocab.encode(self.tgt_texts[idx], language="tgt")
        return src_ids, tgt_ids, len(src_ids), len(tgt_ids)


def collate_fn(batch):
    src_seq, tgt_seq, src_lens, tgt_lens = zip(*batch)
    # pad_sequence expects list of tensors (len varies)
    src_padded = pad_sequence(src_seq, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_seq, batch_first=True, padding_value=0)
    return src_padded, tgt_padded, torch.tensor(src_lens), torch.tensor(tgt_lens)
