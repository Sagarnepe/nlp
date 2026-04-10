import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharDataset:
    def __init__(self, text: str, seq_len: int = 64, val_frac: float = 0.1):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        split = int(len(data) * (1 - val_frac))

        self.train_data = data[:split]
        self.val_data = data[split:]
        self.seq_len = seq_len

    def get_batch(self, split: str = "train", batch_size: int = 64):
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= self.seq_len + 1:
            raise ValueError("Text is too short for the chosen sequence length.")

        starts = np.random.randint(0, len(data) - self.seq_len - 1, size=batch_size)

        x = np.stack([data[s:s + self.seq_len] for s in starts])
        y = np.stack([data[s + 1:s + self.seq_len + 1] for s in starts])

        x = torch.tensor(x, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        return x, y


class CharRNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_size: int = 128,
        rnn_type: str = "lstm"
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        rnn_type = rnn_type.lower()

        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be one of: rnn, gru, lstm")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


@torch.no_grad()
def estimate_loss(model, dataset, batch_size=64, eval_steps=20):
    model.eval()
    stats = {}

    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_steps):
            xb, yb = dataset.get_batch(split=split, batch_size=batch_size)
            logits, _ = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, dataset.vocab_size),
                yb.reshape(-1)
            )
            losses.append(loss.item())
        stats[split] = float(np.mean(losses))

    model.train()
    return stats


@torch.no_grad()
def generate_text(model, dataset, start_text, max_new_chars=300, temperature=1.0):
    model.eval()

    for ch in start_text:
        if ch not in dataset.stoi:
            raise ValueError(f"Character {repr(ch)} not found in vocabulary.")

    idx = torch.tensor(
        [[dataset.stoi[ch] for ch in start_text]],
        dtype=torch.long,
        device=device
    )

    hidden = None

    # Warm up on the prompt
    _, hidden = model(idx, hidden)

    for _ in range(max_new_chars):
        last_char = idx[:, -1:]
        logits, hidden = model(last_char, hidden)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx], dim=1)

    out = "".join(dataset.itos[i] for i in idx[0].tolist())
    return out


def save_loss_plot(train_losses, val_losses, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, required=True, help="Path to plain text file")
    parser.add_argument("--out_dir", type=str, default="runs/char_rnn")
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature_values", type=float, nargs="+", default=[0.7, 1.0, 1.2])
    parser.add_argument("--sample_chars", type=int, default=300)
    parser.add_argument("--start_text", type=str, default="The ")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = Path(args.text_path).read_text(encoding="utf-8")

    dataset = CharDataset(text=text, seq_len=args.seq_len, val_frac=0.1)

    model = CharRNNModel(
        vocab_size=dataset.vocab_size,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        rnn_type=args.rnn_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    print(f"Using device: {device}")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Train chars: {len(dataset.train_data)}")
    print(f"Val chars: {len(dataset.val_data)}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        for _ in range(args.steps_per_epoch):
            xb, yb = dataset.get_batch(split="train", batch_size=args.batch_size)

            # Teacher forcing:
            # input is the true previous character sequence
            logits, _ = model(xb)

            loss = F.cross_entropy(
                logits.reshape(-1, dataset.vocab_size),
                yb.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        stats = estimate_loss(
            model,
            dataset,
            batch_size=args.batch_size,
            eval_steps=20
        )

        train_losses.append(stats["train"])
        val_losses.append(stats["val"])

        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {stats['train']:.4f} | "
            f"val loss: {stats['val']:.4f}"
        )

    # Save loss curve
    save_loss_plot(
        train_losses,
        val_losses,
        out_dir / "loss_curves.png"
    )

    # Save samples
    with open(out_dir / "samples.txt", "w", encoding="utf-8") as f:
        for temp in args.temperature_values:
            sample = generate_text(
                model,
                dataset,
                start_text=args.start_text,
                max_new_chars=args.sample_chars,
                temperature=temp
            )
            f.write(f"{'=' * 20} temperature={temp} {'=' * 20}\n")
            f.write(sample + "\n\n")

    # Save losses as text too
    with open(out_dir / "loss_values.txt", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            f.write(f"{i},{tr:.6f},{va:.6f}\n")

    print(f"\nSaved outputs in: {out_dir}")
    print(" - loss_curves.png")
    print(" - samples.txt")
    print(" - loss_values.txt")


if __name__ == "__main__":
    main()