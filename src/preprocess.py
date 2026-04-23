from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]
    freqs: Dict[str, int]
    pad_idx: int
    unk_idx: int

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def decode(self, idx: int) -> str:
        return self.itos[idx]


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def read_corpus(path: str) -> List[List[str]]:
    sentences: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line.strip())
            if tokens:
                sentences.append(tokens)
    return sentences


def build_vocab(sentences: List[List[str]], min_count: int = 3, max_vocab_size: int | None = None) -> Vocabulary:
    counter = Counter(token for sent in sentences for token in sent)

    filtered = [(word, freq) for word, freq in counter.items() if freq >= min_count]
    filtered.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        filtered = filtered[:max_vocab_size]

    itos = [PAD_TOKEN, UNK_TOKEN] + [word for word, _ in filtered]
    stoi = {word: idx for idx, word in enumerate(itos)}
    freqs = {word: freq for word, freq in filtered}

    return Vocabulary(
        stoi=stoi,
        itos=itos,
        freqs=freqs,
        pad_idx=stoi[PAD_TOKEN],
        unk_idx=stoi[UNK_TOKEN],
    )


def encode_sentences(sentences: List[List[str]], vocab: Vocabulary) -> List[List[int]]:
    return [[vocab.encode(token) for token in sent] for sent in sentences]


def prepare_corpus(
    path: str,
    min_count: int = 3,
    max_vocab_size: int | None = None,
) -> Tuple[List[List[int]], Vocabulary]:
    sentences = read_corpus(path)
    vocab = build_vocab(sentences, min_count=min_count, max_vocab_size=max_vocab_size)
    encoded = encode_sentences(sentences, vocab)
    return encoded, vocab
