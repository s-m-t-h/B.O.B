"""
Trains Bob on the data produced by prepare_data.py.

Run:
    python train.py

Everything below is tuned to train in minutes on a single consumer
NVIDIA GPU on a small text file (a few MB). Scale n_embd/n_layer/n_head
up if you have more data and more patience.
"""

import pickle
import time
import torch

from model import BobGPT

# ---------------------------------------------------------------------------
# Hyperparameters -- the knobs that matter and why
# ---------------------------------------------------------------------------
batch_size = 64        # sequences per training step
block_size = 128       # context length: how many previous tokens the model sees
n_embd = 128            # embedding dimension (model "width")
n_head = 4              # number of attention heads (n_embd must be divisible by this)
n_layer = 4              # number of transformer blocks (model "depth")
dropout = 0.1            # regularization: randomly zero some activations during training
learning_rate = 3e-4      # step size for the optimizer
max_iters = 3000          # total training steps
eval_interval = 250       # how often to check validation loss
eval_iters = 50           # batches averaged for each validation estimate
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Load data prepared by prepare_data.py
# ---------------------------------------------------------------------------
train_data = torch.load("data/train.pt")
val_data = torch.load("data/val.pt")
with open("data/meta.pkl", "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]


def get_batch(split: str):
    """
    Samples a batch of random (input, target) sequence pairs.
    target is input shifted by one position -- that's the entire
    "next token prediction" objective in code form.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Averages loss over several batches for a less noisy readout."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Build model + optimizer
# ---------------------------------------------------------------------------
model = BobGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
start = time.time()
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start
        print(
            f"step {step:5d} | train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | {elapsed:.0f}s elapsed"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
torch.save(
    {
        "model_state": model.state_dict(),
        "config": dict(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
        ),
    },
    "bob_checkpoint.pt",
)
print("Saved bob_checkpoint.pt")
