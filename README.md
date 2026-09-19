# B.O.B
An ai project I am creating named B.O.B, This is still a work in progress and is completely experimental.

A minimal, heavily-commented implementation of a GPT-style transformer
language model, in the spirit of Andrej Karpathy's nanoGPT/makemore.
Small enough to read end-to-end and actually understand; real enough
that it learns and generates text.

**Set expectations correctly**: with a few MB of text and a few million
parameters, Bob will learn *style and local structure* (spelling, grammar,
recurring phrases) but will not have broad world knowledge, reasoning, or
instruction-following. Those emerge at scales of billions of parameters
and hundreds of billions of training tokens, trained for months on
thousands of GPUs, plus a further stage (RLHF / instruction tuning) that
this project doesn't include. This is a real, working demonstration of the
*mechanism* — not a competitor to production models.

## How it works, in order

1. **Tokenization** (`prepare_data.py`): your text file is split into
   individual characters, each assigned an integer ID. This is the
   simplest tokenizer possible — production models use subword tokenizers
   (BPE) so sequences are shorter and more semantically meaningful, but
   character-level is the clearest way to see the whole pipeline.

2. **Architecture** (`model.py`): token IDs → embeddings → N transformer
   blocks (causal self-attention + feedforward, each with residual
   connections and LayerNorm) → final linear layer producing a probability
   distribution over the next character. Read this file — every class maps
   directly to a concept in the theory below.

3. **Training** (`train.py`): repeatedly samples random chunks of your
   text, asks the model to predict each next character, and adjusts
   weights via backpropagation to reduce the error (cross-entropy loss).
   That single objective — next-token prediction — is all a GPT is ever
   trained to do.

4. **Generation** (`chat.py`): feed in a prompt, the model produces a
   probability distribution over the next character, you sample from it,
   append the result, and repeat. This is autoregressive generation.

## The core mechanism: self-attention

This is the one idea worth really internalizing. For a sequence of
tokens, each token produces three vectors:

- **Query** — what this token is looking for
- **Key** — what this token offers, as a label
- **Value** — what this token offers, as content

A token's Query is dot-producted against every other token's Key to get
similarity scores. Those scores are softmaxed into weights (summing to 1),
and the token's new representation is the weighted sum of all Values,
using those weights. High Query·Key similarity means "pay more attention
to this token's Value."

The **causal mask** forces token *i* to only attend to tokens `0..i` —
otherwise the model could "cheat" by looking at the very token it's
supposed to predict.

**Multi-head** just means doing this whole process several times in
parallel with smaller vectors, so different heads can specialize, then
concatenating the results back together.

Everything else in the architecture (embeddings, feedforward layers,
LayerNorm, residual connections) exists to make this mechanism trainable
and expressive when stacked many layers deep.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Get some training text.** Character-level models want a stylistically
   consistent corpus — a book, a collection of someone's writing, song
   lyrics, etc. A few hundred KB is enough to see it learn something; a
   few MB is better. Save it as `data/input.txt`.

   (A classic starter dataset is the "Tiny Shakespeare" text file — search
   for `tinyshakespeare` / `input.txt karpathy char-rnn` and drop it in
   `data/input.txt` if you want a known-good baseline before using your
   own text.)

2. **Prepare it:**
   ```bash
   python prepare_data.py data/input.txt
   ```

3. **Train:**
   ```bash
   python train.py
   ```
   Watch `train loss` and `val loss` — both should steadily decrease. If
   val loss starts rising while train loss keeps falling, the model is
   overfitting (memorizing rather than generalizing) — reduce `max_iters`
   or add more data.

4. **Chat with it:**
   ```bash
   python chat.py
   ```

## Tuning knobs (in `train.py`)

| Parameter | Effect |
|---|---|
| `n_embd` | Model width. Bigger = more capacity, slower, more data needed. |
| `n_layer` | Model depth. More layers = more abstraction, harder to train. |
| `n_head` | Parallel attention "perspectives." Must divide `n_embd` evenly. |
| `block_size` | Context window — how far back the model can "see." |
| `batch_size` | Sequences per step. Bigger = smoother gradients, more GPU memory. |
| `learning_rate` | Step size. Too high = unstable; too low = slow. |
| `max_iters` | Training length. Watch val loss rather than guessing a number. |

Defaults (128 embd, 4 heads, 4 layers, ~a few million params) should train
in well under 10 minutes on a modern NVIDIA GPU for a few-MB text file.

## Where to go from here

- **Word/subword tokenization** instead of characters — shorter sequences,
  usually better results, more setup (train a BPE tokenizer, e.g. with
  the `tokenizers` library).
- **Scale up** `n_embd`/`n_layer`/`n_head` and train on more data — this
  is literally what separates Bob from a frontier model: more of the same
  mechanism, vastly more parameters and data.
- **Instruction tuning** — format training data as
  prompt/response pairs and fine-tune on that, so it responds to
  instructions instead of just continuing text style.
- **Mixed precision / larger batches** — `torch.cuda.amp` for faster
  training if you scale up.

## Files

- `model.py` — architecture (read this first)
- `prepare_data.py` — text → tokenized train/val tensors
- `train.py` — training loop
- `chat.py` — interactive generation
- `data/` — put your `input.txt` here
