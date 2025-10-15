from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from cortex.stacks import CortexStack
from model import SequenceClassifier
from stacks import STACKS, StackSpec
from synthetic_datasets import DelayedRecallDataset, DyckDataset, MajorityDataset, MajorityHeadPadDataset
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
        delay = 512

        def _splits():
            return DelayedRecallDataset.splits(num_samples=num_samples, delay=delay, seed=seed)

        return TaskSpec(name=task, make_splits=_splits, vocab_size=3, n_classes=2)

    if task == "majority":
        length = 1024

        def _splits():
            return MajorityDataset.splits(num_samples=num_samples, length=length, seed=seed)

        return TaskSpec(name=task, make_splits=_splits, vocab_size=3, n_classes=2)

    if task == "majority_headpad":
        length = 1024
        tail_pad_len = 256  # choose chunk-size=256 to make last chunk all PAD

        def _splits():
            return MajorityHeadPadDataset.splits(
                num_samples=num_samples, length=length, tail_pad_len=tail_pad_len, seed=seed
            )

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
    chunk_size: int | None = None,
    rtu_disable_traces_last_chunk: bool = False,
    reset_state_before_last_chunk: bool = False,
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

    # Generic chunked processing (TBPTT on last chunk) for any stack when enabled
    chunk_enabled = bool(chunk_size and chunk_size > 0)
    if chunk_enabled:
        logging.info("chunked processing enabled: chunk_size=%d", int(chunk_size))

    def _zero_rtu_traces_in_state(state) -> None:
        """Zero Axon eligibility traces wherever they appear in the nested state.

        Supports both direct AxonCell states (under a block key) and AxonLayer-managed
        substates nested under groups like 'axon', 'slstm', 'mlstm', 'mlstm_qkv', etc.
        """
        if state is None or not hasattr(state, "keys"):
            return

        trace_keys = (
            "E_nu_c1",
            "E_nu_c2",
            "E_th_c1",
            "E_th_c2",
            "E_w1_c1",
            "E_w1_c2",
            "E_w2_c1",
            "E_w2_c2",
        )

        scanned = 0
        zeroed = 0

        def _recurse(td, path: str) -> None:
            if td is None or not hasattr(td, "keys"):
                return
            try:
                td_keys = list(td.keys())
            except Exception:
                td_keys = []

            # If this TensorDict looks like an Axon trace container, zero its traces
            has_explicit_traces = any(k in td_keys for k in trace_keys)
            looks_like_axon_state = ("hc1" in td_keys) and ("hc2" in td_keys)
            if has_explicit_traces or looks_like_axon_state:
                nonlocal scanned, zeroed
                scanned += 1
                try:
                    pre_stats = []
                    # Collect both explicit and generic E_* traces for logging
                    for name in td_keys:
                        if (name in trace_keys or name.startswith("E_")) and torch.is_tensor(td.get(name)):
                            t = td.get(name)
                            pre_stats.append((name, float(t.norm().item())))
                    if pre_stats:
                        msg = ", ".join(f"{n}:l2={v:.3e}" for n, v in pre_stats)
                        logging.debug("[traces] before-zero path=%s %s", path, msg)
                except Exception:
                    pass
                # Zero any known or generic E_* trace tensors
                for name in list(td_keys):
                    if (name in trace_keys or name.startswith("E_")) and torch.is_tensor(td.get(name)):
                        with torch.no_grad():
                            t = td.get(name)
                            td[name] = torch.zeros_like(t)
                zeroed += 1
                try:
                    post_nonzero = []
                    for name in td_keys:
                        if (name in trace_keys or name.startswith("E_")) and torch.is_tensor(td.get(name)):
                            t = td.get(name)
                            if float(t.abs().sum().item()) != 0.0:
                                post_nonzero.append(name)
                    if post_nonzero:
                        logging.warning("[traces] non-zero after zeroing path=%s keys=%s", path, post_nonzero)
                    else:
                        logging.debug("[traces] after-zero path=%s all-zero", path)
                except Exception:
                    pass

            # Recurse into children TensorDicts
            for k in td_keys:
                try:
                    child = td.get(k)
                except Exception:
                    child = None
                if hasattr(child, "keys"):
                    _recurse(child, f"{path}/{k}")

            # No architecture-specific fallbacks: recursion + generic E_* detection handles all cases

        # Start recursion from each top-level block key
        try:
            top_keys = list(state.keys())
        except Exception:
            top_keys = []
        for k in top_keys:
            try:
                sub = state.get(k)
            except Exception:
                sub = None
            if hasattr(sub, "keys"):
                _recurse(sub, k)
        logging.debug("[traces] summary: scanned=%d zeroed=%d", scanned, zeroed)

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
            if chunk_enabled:
                # Stream in chunks; detach state between chunks (TBPTT on last chunk)
                T = seq.size(1)
                C = int(chunk_size)  # type: ignore[arg-type]
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
                # Optionally disable RTU traces at last chunk for sanity checks (no-op for non-RTU stacks)
                if rtu_disable_traces_last_chunk and state is not None:
                    _zero_rtu_traces_in_state(state)
                # Optionally drop the carried state entirely before final chunk
                if reset_state_before_last_chunk:
                    state = None
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
                and model.stack.blocks[0].cell.__class__.__name__ == "AxonsCell"
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
    parser.add_argument("--task", choices=["delayed_recall", "majority", "majority_headpad", "dyck"], required=True)
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
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, process sequences in chunks of this many timesteps and detach state between chunks "
            "(TBPTT on the last chunk only). Applies to any stack."
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
    parser.add_argument(
        "--reset-state-before-last-chunk",
        action="store_true",
        help=(
            "Diagnostic: set state=None immediately before the final gradient-bearing chunk when chunking is on. "
            "Removes information propagated from earlier chunks; majority and delayed_recall should drop toward chance."
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
            chunk_size=args.chunk_size if args.chunk_size > 0 else None,
            rtu_disable_traces_last_chunk=args.rtu_disable_traces_last_chunk,
            reset_state_before_last_chunk=args.reset_state_before_last_chunk,
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
