from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import CBOWDataset, SkipGramDataset
from models import CBOWModel, SkipGramModel
from preprocess import prepare_corpus
from utils import AverageMeter, ensure_dir, get_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CBOW or Skip-gram word2vec models in PyTorch.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to plain text corpus.")
    parser.add_argument("--model_type", type=str, choices=["cbow", "skipgram"], required=True)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_prefix", type=str, default=None)
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace):
    encoded_sentences, vocab = prepare_corpus(
        path=args.data_path,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
    )

    if args.model_type == "cbow":
        full_dataset = CBOWDataset(encoded_sentences, window_size=args.window_size, pad_idx=vocab.pad_idx)
    else:
        full_dataset = SkipGramDataset(encoded_sentences, window_size=args.window_size)

    if len(full_dataset) < 10:
        raise ValueError("Dataset too small after preprocessing. Use a larger corpus or lower min_count.")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, vocab


def create_model(args: argparse.Namespace, vocab_size: int, pad_idx: int):
    if args.model_type == "cbow":
        return CBOWModel(vocab_size=vocab_size, embedding_dim=args.embedding_dim, pad_idx=pad_idx)
    return SkipGramModel(vocab_size=vocab_size, embedding_dim=args.embedding_dim)


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True) -> float:
    meter = AverageMeter()
    model.train(train)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        if train:
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), n=inputs.size(0))

    return meter.avg


def plot_losses(train_losses, val_losses, output_path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    ensure_dir("results/checkpoints")
    ensure_dir("results/figures")
    ensure_dir("results/embeddings")

    train_loader, val_loader, vocab = build_dataloaders(args)
    model = create_model(args, vocab_size=len(vocab), pad_idx=vocab.pad_idx).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    prefix = args.save_prefix or f"{args.model_type}_d{args.embedding_dim}_w{args.window_size}"
    best_ckpt_path = f"results/checkpoints/best_{prefix}.pt"
    last_ckpt_path = f"results/checkpoints/last_{prefix}.pt"
    plot_path = f"results/figures/{prefix}_loss.png"
    config_path = f"results/checkpoints/{prefix}_config.json"

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    total_start = perf_counter()

    config: Dict[str, object] = vars(args).copy()
    config["vocab_size"] = len(vocab)
    save_json(config, config_path)

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = perf_counter() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={elapsed:.2f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": {
                "stoi": vocab.stoi,
                "itos": vocab.itos,
                "freqs": vocab.freqs,
                "pad_idx": vocab.pad_idx,
                "unk_idx": vocab.unk_idx,
            },
            "config": config,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        save_checkpoint(checkpoint, last_ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint, best_ckpt_path)

    total_time = perf_counter() - total_start
    plot_losses(train_losses, val_losses, plot_path)
    print(f"Training finished in {total_time:.2f} seconds.")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
