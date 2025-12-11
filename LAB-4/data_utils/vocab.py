# data_utils/vocab.py
import re
import os
import pickle
import torch

class Vocab:
    def __init__(self, src_sentences, tgt_sentences, cache_path=None):
        """
        src_sentences, tgt_sentences: list of raw strings
        cache_path: optional path to save/load vocab to speed up
        """
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            (self.src_itos, self.tgt_itos, self.src_stoi, self.tgt_stoi) = data
        else:
            self.src_itos, self.src_stoi = self._build_from_sentences(src_sentences)
            self.tgt_itos, self.tgt_stoi = self._build_from_sentences(tgt_sentences)
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump((self.src_itos, self.tgt_itos, self.src_stoi, self.tgt_stoi), f)

    def preprocess(self, text):
        if isinstance(text, list):
            return text
        if text is None:
            return []

        # remove BOM if exists
        text = text.replace("\ufeff", "")

        # lowercase
        text = text.lower().strip()

        # fast punctuation split without heavy regex
        punct = ".,!?\"'():;/\\-"
        for p in punct:
            text = text.replace(p, f" {p} ")

        # normalize spaces
        text = " ".join(text.split())

        return text.split()


    def _build_from_sentences(self, sentences):
        words = set()
        for s in sentences:
            toks = self.preprocess(s)
            words.update(toks)
        # do not sort if huge; sorting fine for small vocab
        itos = self.special_tokens + sorted(words)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return itos, stoi

    @property
    def total_src_tokens(self): return len(self.src_itos)
    @property
    def total_tgt_tokens(self): return len(self.tgt_itos)

    def encode(self, text, language="src"):
        toks = self.preprocess(text)
        stoi = self.src_stoi if language == "src" else self.tgt_stoi
        ids = [self.bos_idx] + [stoi.get(t, self.unk_idx) for t in toks] + [self.eos_idx]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids, language="tgt"):
        itos = self.src_itos if language == "src" else self.tgt_itos
        words = []
        for i in ids:
            if i in (self.pad_idx, self.bos_idx, self.eos_idx):
                continue
            words.append(itos[int(i)])
        return " ".join(words)
