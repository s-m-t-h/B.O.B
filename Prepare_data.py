import sys
import pickle
import torch

def main():
    if len(sys.argv) != 2:
        print("Usage: python prepare_data.py <path_to_text_file>")
        sys.exit(1)

    text_path = sys.argv[1]
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters from {text_path}")

    # Build the vocabulary: every unique character that appears.
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)

    # 90/10 train/val split. The val set tells us whether the model is
    # generalizing or just memorizing the training text.
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    torch.save(train_data, "data/train.pt")
    torch.save(val_data, "data/val.pt")
    with open("data/meta.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, f)

    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    print("Saved data/train.pt, data/val.pt, data/meta.pkl")


if __name__ == "__main__":
    main()
