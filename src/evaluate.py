from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import CBOWModel, SkipGramModel
from utils import load_checkpoint, get_device


class VocabularyView:
    def __init__(self, vocab_dict: Dict) -> None:
        self.stoi = vocab_dict["stoi"]
        self.itos = vocab_dict["itos"]
        self.freqs = vocab_dict["freqs"]
        self.pad_idx = vocab_dict["pad_idx"]
        self.unk_idx = vocab_dict["unk_idx"]

    def encode(self, word: str) -> int:
        return self.stoi.get(word, self.unk_idx)

    def decode(self, idx: int) -> str:
        return self.itos[idx]

    def __len__(self) -> int:
        return len(self.itos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained word2vec models.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--query_word", type=str, default=None, help="Word for nearest-neighbor search.")
    parser.add_argument(
        "--analogy",
        type=str,
        default=None,
        help='Three words in the form "A B C" for A:B::C:? analogy.',
    )
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def load_model_and_vocab(checkpoint_path: str, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    vocab = VocabularyView(checkpoint["vocab"])

    if config["model_type"] == "cbow":
        model = CBOWModel(vocab_size=len(vocab), embedding_dim=config["embedding_dim"], pad_idx=vocab.pad_idx)
    else:
        model = SkipGramModel(vocab_size=len(vocab), embedding_dim=config["embedding_dim"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, config


@torch.no_grad()
def get_normalized_embeddings(model) -> torch.Tensor:
    embeddings = model.embeddings.weight.detach()
    return F.normalize(embeddings, p=2, dim=1)


@torch.no_grad()
def nearest_neighbors(word: str, model, vocab: VocabularyView, top_k: int = 10) -> List[Tuple[str, float]]:
    idx = vocab.encode(word)
    if idx == vocab.unk_idx and word not in vocab.stoi:
        raise ValueError(f"Word '{word}' not found in vocabulary.")

    normalized = get_normalized_embeddings(model)
    query = normalized[idx]
    scores = torch.matmul(normalized, query)
    scores[idx] = -1.0
    if vocab.pad_idx < len(scores):
        scores[vocab.pad_idx] = -1.0
    if vocab.unk_idx < len(scores):
        scores[vocab.unk_idx] = -1.0

    top_scores, top_indices = torch.topk(scores, k=top_k)
    return [(vocab.decode(i.item()), s.item()) for i, s in zip(top_indices, top_scores)]


@torch.no_grad()
def analogy(a: str, b: str, c: str, model, vocab: VocabularyView, top_k: int = 10) -> List[Tuple[str, float]]:
    words = [a, b, c]
    ids = []
    for word in words:
        idx = vocab.encode(word)
        if idx == vocab.unk_idx and word not in vocab.stoi:
            raise ValueError(f"Word '{word}' not found in vocabulary.")
        ids.append(idx)

    normalized = get_normalized_embeddings(model)
    target_vec = normalized[ids[1]] - normalized[ids[0]] + normalized[ids[2]]
    target_vec = F.normalize(target_vec.unsqueeze(0), p=2, dim=1).squeeze(0)

    scores = torch.matmul(normalized, target_vec)
    for idx in ids + [vocab.pad_idx, vocab.unk_idx]:
        if idx < len(scores):
            scores[idx] = -1.0

    top_scores, top_indices = torch.topk(scores, k=top_k)
    return [(vocab.decode(i.item()), s.item()) for i, s in zip(top_indices, top_scores)]


def main() -> None:
    args = parse_args()
    device = get_device()
    model, vocab, config = load_model_and_vocab(args.checkpoint_path, device)

    print("Loaded model:")
    print(config)
    print()

    if args.query_word:
        print(f"Nearest neighbors for '{args.query_word}':")
        for word, score in nearest_neighbors(args.query_word.lower(), model, vocab, top_k=args.top_k):
            print(f"  {word:20s} {score:.4f}")
        print()

    if args.analogy:
        parts = args.analogy.lower().split()
        if len(parts) != 3:
            raise ValueError("Analogy must contain exactly three words: 'A B C'.")
        a, b, c = parts
        print(f"Analogy: {a}:{b}::{c}:?")
        for word, score in analogy(a, b, c, model, vocab, top_k=args.top_k):
            print(f"  {word:20s} {score:.4f}")

    if not args.query_word and not args.analogy:
        print("No evaluation action provided. Use --query_word and/or --analogy.")


if __name__ == "__main__":
    main()
