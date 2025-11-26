import torch
import string
from torch.utils.data import Dataset
from collections import Counter


class Vocab:
    def __init__(self, texts, labels):
        # build vocab from all training texts
        counter = Counter()

        for txt in texts:
            txt = self.preprocess_sentence(txt)
            counter.update(txt.split())

        # keep top 5000 words
        most_common = counter.most_common(5000)

        self.w2i = {"PAD": 0, "UNK": 1}
        for i, (word, _) in enumerate(most_common, start=2):
            self.w2i[word] = i

        self.i2w = {i: w for w, i in self.w2i.items()}

        # label mapping
        self.l2i = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.i2l = {v: k for k, v in self.l2i.items()}

    @property
    def n_labels(self):
        return len(self.l2i)

    def preprocess_sentence(self, sentence: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return sentence.lower().translate(translator)

    def encode_sentence(self, sentence: str):
        MAX_LEN = 50
        sentence = self.preprocess_sentence(sentence)
        words = sentence.split()[:MAX_LEN]

        ids = [self.w2i.get(w, 1) for w in words]  # 1 = UNK
        return torch.tensor(ids, dtype=torch.long)


class VSFCDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.vocab.encode_sentence(self.texts[idx])
        y = self.vocab.l2i[self.labels[idx]]
        return x, y


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    return padded, torch.tensor(labels), lengths
