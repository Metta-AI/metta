from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
import logging
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure local cortex package is importable when executed from repo root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _ROOT not in os.sys.path:
    os.sys.path.insert(0, _ROOT)

from cortex.stacks import CortexStack  # noqa: E402

# Support running as a script (no package parent)
try:
    from .model import SequenceClassifier  # type: ignore
    from .stacks import STACKS, StackSpec  # type: ignore
    from .synthetic_datasets import (  # type: ignore
        DelayedRecallDataset,
        DyckDataset,
        MajorityDataset,
    )
except Exception:
    import sys as _sys

    _EVAL_ROOT = os.path.dirname(__file__)
    if _EVAL_ROOT not in _sys.path:
        _sys.path.insert(0, _EVAL_ROOT)
    from model import SequenceClassifier  # type: ignore
    from stacks import STACKS, StackSpec  # type: ignore
    from synthetic_datasets import (  # type: ignore
        DelayedRecallDataset,
        DyckDataset,
        MajorityDataset,
    )


def _ensure_disjoint_splits(train_ds, val_ds, test_ds) -> None:
    """Verify train/val/test splits are disjoint by global sample ids.

    All synthetic datasets in this harness expose an `ids` attribute that
    corresponds to the original generation indices. We use sets of these ids
    to assert there is no overlap between splits. If `ids` is missing, we
    log a warning and skip the check.
    """
    def _get_ids(ds):
        return getattr(ds, "ids", None)

    train_ids = _get_ids(train_ds)
    val_ids = _get_ids(val_ds)
    test_ids = _get_ids(test_ds)

    if any(ids is None for ids in (train_ids, val_ids, test_ids)):
        logging.warning("split-disjointness check skipped: `ids` attribute missing on dataset")
        return

    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    assert s_tr.isdisjoint(s_va), "train/val split leak detected"
    assert s_tr.isdisjoint(s_te), "train/test split leak detected"
    assert s_va.isdisjoint(s_te), "val/test split leak detected"


@dataclass
class TaskSpec:
    name: str
    make_splits: Callable[[], Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]]
    vocab_size: int
    n_classes: int


def make_task(task: str, *, num_samples: int, seed: int) -> TaskSpec:
    if task == "delayed_recall":
        delay = 50

        def _splits():
            return DelayedRecallDataset.splits(num_samples=num_samples, delay=delay, seed=seed)

        return TaskSpec(name=task, make_splits=_splits, vocab_size=3, n_classes=2)

    if task == "majority":
        length = 100

        def _splits():
            return MajorityDataset.splits(num_samples=num_samples, length=length, seed=seed)

        return TaskSpec(name=task, make_splits=_splits, vocab_size=3, n_classes=2)

    if task == "dyck":
        n_pairs = 50

        def _splits():
            return DyckDataset.splits(num_samples=num_samples, n_pairs=n_pairs, seed=seed)

        return TaskSpec(name=task, make_splits=_splits, vocab_size=2, n_classes=2)

    raise ValueError(f"Unknown task: {task}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(
    *,
    stack: CortexStack,
    d_hidden: int,
    task: TaskSpec,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, float]:
    train_ds, val_ds, test_ds = task.make_splits()
    _ensure_disjoint_splits(train_ds, val_ds, test_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SequenceClassifier(stack=stack, vocab_size=task.vocab_size, d_hidden=d_hidden, n_classes=task.n_classes)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def _run_epoch(loader: DataLoader, train: bool) -> Tuple[float, float]:
        model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0
        for seq, labels in loader:
            state = None  # independent sequences; do not carry state across batches during training/eval
            seq = seq.to(device)
            labels = labels.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            logits, state = model(seq, state)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            total_loss += loss.item() * seq.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += seq.size(0)
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(train_loader, train=True)
        val_loss, val_acc = _run_epoch(val_loader, train=False)
        best_val = max(best_val, val_acc)
        current_lr = opt.param_groups[0].get("lr", float("nan"))
        logging.info(
            "epoch=%d lr=%.6f train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            current_lr,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    test_loss, test_acc = _run_epoch(test_loader, train=False)
    return {"val_acc": best_val, "test_acc": test_acc, "test_loss": test_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cortex synthetic evaluations")
    parser.add_argument("--task", choices=["delayed_recall", "majority", "dyck"], required=True)
    parser.add_argument("--stack", choices=["slstm_postup", "mlstm_preup", "xlstm", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Configure logging once
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.getLogger().setLevel(level)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s seed=%d", device, args.seed)

    task = make_task(args.task, num_samples=args.num_samples, seed=args.seed)

    to_run: Dict[str, StackSpec]
    if args.stack == "all":
        to_run = STACKS
    else:
        to_run = {args.stack: STACKS[args.stack]}

    results: Dict[str, Dict[str, float]] = {}
    for name, spec in to_run.items():
        logging.info("starting stack=%s d_hidden=%d", name, spec.d_hidden)
        stack = spec.builder().to(device)
        metrics = train_one(
            stack=stack,
            d_hidden=spec.d_hidden,
            task=task,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        results[name] = metrics
        logging.info("finished stack=%s val_acc=%.4f test_acc=%.4f test_loss=%.4f", name, metrics["val_acc"], metrics["test_acc"], metrics["test_loss"])

    # Pretty print
    logging.info("summary task=%s results=%s", task.name, results)


if __name__ == "__main__":
    main()
