from __future__ import annotations

import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        # context_words: [batch, 2*window]
        embedded = self.embeddings(context_words)  # [batch, context, dim]

        mask = (context_words != self.embeddings.padding_idx).unsqueeze(-1)
        masked_embedded = embedded * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        context_vector = masked_embedded.sum(dim=1) / lengths

        logits = self.output(context_vector)
        return logits


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words: torch.Tensor) -> torch.Tensor:
        # center_words: [batch]
        embedded = self.embeddings(center_words)  # [batch, dim]
        logits = self.output(embedded)
        return logits
