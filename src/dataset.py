from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class CBOWDataset(Dataset):
    def __init__(self, encoded_sentences: Sequence[Sequence[int]], window_size: int, pad_idx: int) -> None:
        self.samples: List[Tuple[List[int], int]] = []
        self.window_size = window_size
        self.pad_idx = pad_idx
        self.context_size = 2 * window_size

        for sent in encoded_sentences:
            if len(sent) < 2:
                continue
            for i in range(len(sent)):
                context = []
                for j in range(i - window_size, i + window_size + 1):
                    if j == i:
                        continue
                    if 0 <= j < len(sent):
                        context.append(sent[j])
                    else:
                        context.append(pad_idx)
                target = sent[i]
                self.samples.append((context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SkipGramDataset(Dataset):
    def __init__(self, encoded_sentences: Sequence[Sequence[int]], window_size: int) -> None:
        self.samples: List[Tuple[int, int]] = []

        for sent in encoded_sentences:
            if len(sent) < 2:
                continue
            for i, center_word in enumerate(sent):
                left = max(0, i - window_size)
                right = min(len(sent), i + window_size + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    context_word = sent[j]
                    self.samples.append((center_word, context_word))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        center, context = self.samples[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)
