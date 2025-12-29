import pickle
from collections import Counter

class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        self.pad = "<pad>"
        self.unk = "<unk>"

        self.token2id = {
            self.pad: 0,
            self.unk: 1
        }

        for token, freq in counter.items():
            if freq >= min_freq:
                self.token2id[token] = len(self.token2id)

        self.id2token = {i: t for t, i in self.token2id.items()}

    def encode(self, text):
        return [self.token2id.get(tok, 1) for tok in text.split()]

    def __len__(self):
        return len(self.token2id)
