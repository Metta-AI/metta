Synthetic evaluations for Cortex stacks

This folder contains a lightweight harness to train/evaluate Cortex stacks on a few synthetic sequence tasks (delayed recall, majority vote, Dyck-1). It is intended for quick debugging and comparisons between stack recipes.

Quick start
- CPU/GPU is auto-detected. CUDA is used when available.
- Run one or all stacks against a task:

  - Single stack
    - `python packages/cortex/evaluations/run.py --task delayed_recall --stack slstm_postup`
  - All registered stacks
    - `python packages/cortex/evaluations/run.py --task majority --stack all`

Common flags
- `--task {delayed_recall, majority, dyck}`
- `--stack {slstm_postup, mlstm_preup, xlstm, all}` (auto-discovers new stacks added to `STACKS`)
- `--epochs`, `--batch-size`, `--lr`, `--seed`
- `--log-level {DEBUG, INFO, WARNING, ERROR}` (default: `INFO`)

Add a new stack
- Edit `packages/cortex/evaluations/stacks.py` and register a new entry in `STACKS` following the existing examples.

Notes
- The synthetic dataset implementations are a local copy of `temp/rnn_evaluation/src/synthetic_datasets.py` (only the synthetic tasks are included here).
- This harness keeps things minimal on purpose; it is not aimed at state-of-the-art training, just fast debugging of block/cell wiring and step/sequence parity.

Tasks
- Delayed Recall (T‑Maze)
  - Tokens: {0, 1, 2} where 2 is a filler/PAD token.
  - Sequence: length = delay + 1. The first token is a binary cue bit (0/1); the remaining positions are filler (2).
  - Label: equals the cue bit. The model must store the first token across a variable delay and recall it at the end.
  - Probes: long‑range memory retention, robust state preservation over many timesteps, and resistance to drift when inputs carry no information.
  - Defaults here: delay=50, vocab_size=3, binary classification.

- Majority (Running Majority / Counting)
  - Tokens: {0 (pad), 1 ("+1"), 2 ("‑1")}.
  - Sequence: fixed length L. Each position is non‑zero with probability p; non‑zeros are equally likely 1 or 2.
  - Label: 1 if count(1) > count(2); otherwise 0.
  - Probes: integration over time, stability of additive memory, and balanced credit assignment when signals are sparse.
  - Defaults here: length=100, nonzero_prob=0.3, vocab_size=3, binary classification.

- Dyck‑1 (Balanced Parentheses)
  - Tokens: {0: "(", 1: ")"}.
  - Sequence: length = 2 × n_pairs. Start from a valid balanced sequence; with probability `neg_fraction` flip one random token to make an invalid example.
  - Label: 1 if the sequence is well‑formed (balanced), else 0.
  - Probes: stack‑like behavior (implicit push/pop), nesting depth tracking, and non‑local constraints that depend on the entire prefix history.
  - Defaults here: n_pairs=50, neg_fraction=0.5, vocab_size=2, binary classification.

Modeling assumptions in this harness
- Inputs are integer token IDs; the classifier embeds tokens, runs the chosen Cortex stack, then uses last‑timestep pooling for logits.
- Batches are independent; hidden state is not carried across batches during training/eval in this harness.

Logging
- Uses Python's `logging` (default level INFO) and prints a per-epoch line: `epoch, lr, train_loss/acc, val_loss/acc`.
- A final summary is logged per stack and for the overall run.
