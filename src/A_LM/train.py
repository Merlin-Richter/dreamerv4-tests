import argparse
from dataclasses import asdict
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ModelConfig, Transformer

# ── Dataset ──────────────────────────────────────────────────────────────────


class ShakespeareDataset(Dataset):
    def __init__(self, text: str, seq_len: int, chars: list):
        self.seq_len = seq_len
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint.pt",
        help="Where to save the model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # ── Load Data ───────────────────────────────────────────────────────────

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url, timeout=60).text

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(chars)

    split = int(len(text) * 0.9)
    train_text = text[:split]
    val_text = text[split:]

    train_ds = ShakespeareDataset(train_text, args.seq_len, chars)
    val_ds = ShakespeareDataset(val_text, args.seq_len, chars)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Config & Model ────────────────────────────────────────────────────────

    config = ModelConfig(
        vokab_size=vocab_size,
        embedding_dim=128,
        max_sequence_length=args.seq_len,
        n_heads=8,
        depth=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ─────────────────────────────────────────────────────────

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0, mininterval=1.0)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        for x, y in tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(
                val_loader,
                desc="Val",
                leave=False,
                position=1,
                mininterval=1.0,
            ):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits.view(-1, vocab_size), y.view(-1)).item()

        train_loss = total_loss / len(train_loader)
        val_loss_mean = val_loss / len(val_loader)
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss_mean:.4f}")
        print(
            f"Epoch {epoch + 1} | train loss: {train_loss:.4f} | val loss: {val_loss_mean:.4f}"
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "chars": chars,
        }
        torch.save(payload, args.checkpoint)
        tqdm.write(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
