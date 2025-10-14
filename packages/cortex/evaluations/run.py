from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from cortex.cells.axons import Axons  # runtime type check for gating
from cortex.stacks import CortexStack
from model import SequenceClassifier
from stacks import STACKS, StackSpec
from synthetic_datasets import DelayedRecallDataset, DyckDataset, MajorityDataset
from torch.utils.data import DataLoader


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
        delay = 1024

        def _splits():
            return DelayedRecallDataset.splits(num_samples=num_samples, delay=delay, seed=seed)

        return TaskSpec(name=task, make_splits=_splits, vocab_size=3, n_classes=2)

    if task == "majority":
        length = 1024

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


def _enable_determinism() -> None:
    """Force deterministic behavior where possible (CUDA/cuBLAS/torch)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


def train_one(
    *,
    stack: CortexStack,
    d_hidden: int,
    task: TaskSpec,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    rtu_chunk_size: int | None = None,
    rtu_disable_traces_last_chunk: bool = False,
) -> Dict[str, float]:
    train_ds, val_ds, test_ds = task.make_splits()
    _ensure_disjoint_splits(train_ds, val_ds, test_ds)
    # Ensure identical shuffle order across runs by seeding a dedicated generator
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SequenceClassifier(stack=stack, vocab_size=task.vocab_size, d_hidden=d_hidden, n_classes=task.n_classes)
    model.to(device)

    # Count and log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("model parameters: total=%d trainable=%d", total_params, trainable_params)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def _stack_uses_rtu_stream(s: CortexStack) -> bool:
        """Return True if any block in the stack uses Axons (RTU stream)."""
        for blk in s.blocks:
            # Adapter blocks wrap another block; in our templates we use PreUp directly
            cell = getattr(blk, "cell", None)
            if isinstance(cell, Axons):
                return True
        return False

    rtu_enabled = bool(rtu_chunk_size and rtu_chunk_size > 0 and _stack_uses_rtu_stream(stack))
    if rtu_chunk_size and rtu_chunk_size > 0 and not rtu_enabled:
        logging.warning("--rtu-chunk-size provided but current stack has no RTUStream cell; ignoring chunked streaming")

    def _zero_rtu_traces_in_state(state) -> None:
        """Zero carried streaming traces inside RTUStream cell states in-place.

        This disables boundary-correction terms for the next chunk.
        """
        if state is None:
            return
        try:
            keys = list(state.keys()) if hasattr(state, "keys") else []
        except Exception:
            keys = []
        for bk in keys:
            try:
                bstate = state.get(bk)
            except Exception:
                bstate = None
            if bstate is None or not hasattr(bstate, "keys"):
                continue
            # Axons state (if present) lives under this key
            if "Axons" in bstate.keys():
                try:
                    cstate = bstate.get("Axons")
                except Exception:
                    cstate = None
                if cstate is None or not hasattr(cstate, "keys"):
                    continue
                for name in (
                    "E_nu_c1",
                    "E_nu_c2",
                    "E_th_c1",
                    "E_th_c2",
                    "E_w1_c1",
                    "E_w1_c2",
                    "E_w2_c1",
                    "E_w2_c2",
                ):
                    if name in cstate.keys():
                        t = cstate.get(name)
                        if torch.is_tensor(t):
                            with torch.no_grad():
                                cstate[name] = torch.zeros_like(t)

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
            if rtu_enabled:
                # Stream in chunks; detach state between chunks (TBPTT on last chunk)
                T = seq.size(1)
                C = int(rtu_chunk_size)  # type: ignore[arg-type]
                assert C > 0
                logits = None  # type: ignore[assignment]

                n_full = T // C
                tail = T % C

                # Process preceding chunks without gradients
                if n_full > 0:
                    last_full_to_process = n_full if tail > 0 else max(n_full - 1, 0)
                    for i in range(last_full_to_process):
                        start = i * C
                        end = start + C
                        with torch.no_grad():
                            _out, state = model(seq[:, start:end], state)
                            if state is not None:
                                try:
                                    state = state.detach()
                                except Exception:
                                    state = state.apply(lambda t: t.detach() if torch.is_tensor(t) else t)

                # Final chunk with gradients (tail if present, else last full chunk)
                # Optionally disable traces at last chunk for sanity checks
                if rtu_disable_traces_last_chunk and state is not None:
                    _zero_rtu_traces_in_state(state)
                if tail > 0:
                    start = n_full * C
                    logits, state = model(seq[:, start:], state)
                else:
                    start = (n_full - 1) * C if n_full > 0 else 0
                    logits, state = model(seq[:, start : start + C], state)
            else:
                logits, state = model(seq, state)
            loss = criterion(logits, labels)
            # Optional Axons parity probe: compute grads with both backends on the same minibatch
            # to catch any drift. This is a no-op for optimization; used for diagnostics only.
            if (
                train
                and AXONS_PARITY_PROBE > 0
                and hasattr(model.stack.blocks[0], "cell")
                and model.stack.blocks[0].cell.__class__.__name__ == "Axons"
                and total == 0  # first minibatch only to avoid overhead
                and (EPOCH_IDX % max(AXONS_PARITY_PROBE, 1) == 0)
            ):
                import copy

                model_a = copy.deepcopy(model).to(device)
                model_b = copy.deepcopy(model).to(device)
                # Disable Triton for model_a
                os.environ["CORTEX_DISABLE_TRITON"] = "1"
                opt_a = torch.optim.SGD(model_a.parameters(), lr=0.0)
                opt_a.zero_grad(set_to_none=True)
                logits_a, _ = model_a(seq, state=None)
                loss_a = criterion(logits_a, labels)
                loss_a.backward()
                grads_a = [p.grad.detach().flatten() for p in model_a.parameters() if p.grad is not None]

                # Enable Triton for model_b
                os.environ.pop("CORTEX_DISABLE_TRITON", None)
                opt_b = torch.optim.SGD(model_b.parameters(), lr=0.0)
                opt_b.zero_grad(set_to_none=True)
                logits_b, _ = model_b(seq, state=None)
                loss_b = criterion(logits_b, labels)
                loss_b.backward()
                grads_b = [p.grad.detach().flatten() for p in model_b.parameters() if p.grad is not None]

                if len(grads_a) == len(grads_b):
                    diffs = [torch.max(torch.abs(a - b)).item() for a, b in zip(grads_a, grads_b, strict=False)]
                    max_diff = max(diffs) if diffs else 0.0
                    logging.info(
                        "[parity] max_grad_diff=%.3e (loss_pt=%.6f loss_tr=%.6f)",
                        max_diff,
                        loss_a.item(),
                        loss_b.item(),
                    )
                else:
                    logging.info("[parity] grad list length mismatch: %d vs %d", len(grads_a), len(grads_b))
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
    # Globals for the parity probe inside the epoch loop
    global AXONS_PARITY_PROBE, EPOCH_IDX
    AXONS_PARITY_PROBE = int(os.getenv("AXONS_PARITY_PROBE", "0"))
    EPOCH_IDX = 0
    if AXONS_PARITY_PROBE <= 0:
        # allow CLI flag to control as well
        AXONS_PARITY_PROBE = 0

    for epoch in range(1, epochs + 1):
        EPOCH_IDX = epoch
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
    # Build stack choices dynamically from the registry so new entries in STACKS are auto-discovered.
    stack_choices = sorted(list(STACKS.keys())) + ["all"]
    parser.add_argument("--stack", choices=stack_choices, default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA/torch settings (useful for A/B backend parity).",
    )
    parser.add_argument(
        "--axons-parity-probe",
        type=int,
        default=0,
        help=(
            "If >0 and the stack uses Axons, every N epochs run a no-op parity check on one minibatch "
            "(compute grads with PyTorch and Triton backends back-to-back) and log max grad diff."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument(
        "--rtu-chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, enable streaming in RTU stacks by chunking sequences into this many tokens and detaching state "
            "between chunks (TBPTT on last chunk only). Ignored for non-RTU stacks."
        ),
    )
    parser.add_argument(
        "--rtu-disable-traces-last-chunk",
        action="store_true",
        help=(
            "Sanity check: zero carried RTU streaming traces right before the final gradient-bearing chunk. "
            "Disables boundary-correction terms so delayed_recall should degrade under chunking."
        ),
    )
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
    if args.deterministic:
        _enable_determinism()
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
            rtu_chunk_size=args.rtu_chunk_size if args.rtu_chunk_size > 0 else None,
            rtu_disable_traces_last_chunk=args.rtu_disable_traces_last_chunk,
        )
        results[name] = metrics
        logging.info(
            "finished stack=%s val_acc=%.4f test_acc=%.4f test_loss=%.4f",
            name,
            metrics["val_acc"],
            metrics["test_acc"],
            metrics["test_loss"],
        )

    # Pretty print
    logging.info("summary task=%s results=%s", task.name, results)


if __name__ == "__main__":
    main()
