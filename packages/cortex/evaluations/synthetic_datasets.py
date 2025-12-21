import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

################################################################################
# Utility helpers
################################################################################


def _split_sizes(total: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    """Return number of samples in train/val/test given total and fractions."""
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    n_test = total - n_train - n_val
    assert n_train > 0 and n_val > 0 and n_test > 0, "Fractions must leave >=1 example per split"
    return n_train, n_val, n_test


################################################################################
# 1. Delayed‑Recall (T‑Maze) Dataset
################################################################################


class DelayedRecallDataset(Dataset):
    """Binary delayed‑recall / T‑maze synthetic dataset.

    Each sequence:
        token_0      – cue bit {0,1}
        token_1..L   – filler token (=2)
    Target label      – cue bit (0/1)
    """

    PAD_TOKEN = 2  # value for filler steps

    def __init__(
        self,
        num_samples: int,
        delay: int,
        *,
        seed: int = 0,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.delay = delay
        self.seed = seed

        # pre‑generate full dataset deterministically
        self._seqs, self._labels = self._generate_all(num_samples, delay, seed)

        # slice for (train/val/test)
        end_idx = num_samples if end_idx is None else end_idx
        self._seqs = self._seqs[start_idx:end_idx]
        self._labels = self._labels[start_idx:end_idx]
        # track global sample ids to verify split disjointness
        self.ids = list(range(start_idx, end_idx))

    @staticmethod
    def _generate_all(num_samples: int, delay: int, seed: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        rng = random.Random(seed)
        seqs: List[torch.Tensor] = []
        labels: List[int] = []
        seq_len = delay + 1
        for _ in range(num_samples):
            cue = rng.randint(0, 1)  # 0/1 equally likely
            seq = torch.full((seq_len,), DelayedRecallDataset.PAD_TOKEN, dtype=torch.long)
            seq[0] = cue
            seqs.append(seq)
            labels.append(cue)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return seqs, labels_tensor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self._seqs[idx], self._labels[idx]

    @classmethod
    def splits(
        cls,
        num_samples: int,
        delay: int,
        *,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 0,
    ) -> Tuple["DelayedRecallDataset", "DelayedRecallDataset", "DelayedRecallDataset"]:
        n_train, n_val, _ = _split_sizes(num_samples, train_frac, val_frac)
        train = cls(num_samples, delay, seed=seed, start_idx=0, end_idx=n_train)
        val = cls(num_samples, delay, seed=seed, start_idx=n_train, end_idx=n_train + n_val)
        test = cls(num_samples, delay, seed=seed, start_idx=n_train + n_val)
        return train, val, test


################################################################################
# 2. Running‑Majority (Counting) Dataset
################################################################################


class MajorityDataset(Dataset):
    """Binary majority‑vote dataset (integrator).

    Sequence tokens: {0 (pad), 1 ("+1"), 2 ("‑1")}
    Label = 1  if count(1) > count(2)   else 0.
    """

    PAD, POS, NEG = 0, 1, 2

    def __init__(
        self,
        num_samples: int,
        length: int,
        *,
        nonzero_prob: float = 0.3,
        seed: int = 0,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples, self.length = num_samples, length
        self.nonzero_prob = nonzero_prob
        self.seed = seed

        seqs, labels = self._generate_all(num_samples, length, nonzero_prob, seed)
        end_idx = num_samples if end_idx is None else end_idx
        self._seqs = seqs[start_idx:end_idx]
        self._labels = labels[start_idx:end_idx]
        self.ids = list(range(start_idx, end_idx))

    @staticmethod
    def _generate_all(
        num_samples: int, length: int, nonzero_prob: float, seed: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        seqs: List[torch.Tensor] = []
        labels: List[int] = []
        for _ in range(num_samples):
            # mask deciding which positions carry ±1 vs pad
            mask = torch.rand(length) < nonzero_prob
            signs = torch.where(torch.rand(length) < 0.5, MajorityDataset.NEG, MajorityDataset.POS)
            seq = torch.where(mask, signs, MajorityDataset.PAD)
            pos_cnt = (seq == MajorityDataset.POS).sum().item()
            neg_cnt = (seq == MajorityDataset.NEG).sum().item()
            label = int(pos_cnt > neg_cnt)
            seqs.append(seq.long())
            labels.append(label)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return seqs, labels_tensor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self._seqs[idx], self._labels[idx]

    @classmethod
    def splits(
        cls,
        num_samples: int,
        length: int,
        *,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        nonzero_prob: float = 0.3,
        seed: int = 0,
    ) -> Tuple["MajorityDataset", "MajorityDataset", "MajorityDataset"]:
        n_train, n_val, _ = _split_sizes(num_samples, train_frac, val_frac)
        train = cls(
            num_samples,
            length,
            nonzero_prob=nonzero_prob,
            seed=seed,
            start_idx=0,
            end_idx=n_train,
        )
        val = cls(
            num_samples,
            length,
            nonzero_prob=nonzero_prob,
            seed=seed,
            start_idx=n_train,
            end_idx=n_train + n_val,
        )
        test = cls(
            num_samples,
            length,
            nonzero_prob=nonzero_prob,
            seed=seed,
            start_idx=n_train + n_val,
        )
        return train, val, test


################################################################################
# 3. Balanced‑Parentheses (Dyck‑1) Dataset
################################################################################


class DyckDataset(Dataset):
    """Dyck‑1 balanced‑parentheses membership (binary).

    Token mapping: 0 = "(", 1 = ")". Label: 1 if sequence is well‑formed, else 0.
    n_pairs gives maximum nesting length; sequence length = 2 * n_pairs.
    """

    OPEN, CLOSE = 0, 1

    def __init__(
        self,
        num_samples: int,
        n_pairs: int,
        *,
        seed: int = 0,
        neg_fraction: float = 0.5,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples, self.n_pairs = num_samples, n_pairs
        self.seed = seed
        seqs, labels = self._generate_all(num_samples, n_pairs, seed, neg_fraction)
        end_idx = num_samples if end_idx is None else end_idx
        self._seqs = seqs[start_idx:end_idx]
        self._labels = labels[start_idx:end_idx]
        self.ids = list(range(start_idx, end_idx))

    @staticmethod
    def _gen_balanced(gen: random.Random, n_pairs: int) -> List[int]:
        """Sample a uniformly random Dyck-1 string of length 2*n_pairs.

        Enforces both constraints at each step:
        - Never close below zero balance (forces open when balance==0)
        - Never open beyond n_pairs (forces close when opens==n_pairs)

        By construction, the sequence ends with balance==0 and has exactly
        n_pairs opens and n_pairs closes.
        """
        seq: List[int] = []
        opens = 0
        closes = 0
        for _ in range(2 * n_pairs):
            if opens == n_pairs:
                seq.append(DyckDataset.CLOSE)
                closes += 1
            elif closes == opens:
                seq.append(DyckDataset.OPEN)
                opens += 1
            else:
                if gen.random() < 0.5:
                    seq.append(DyckDataset.OPEN)
                    opens += 1
                else:
                    seq.append(DyckDataset.CLOSE)
                    closes += 1
        return seq

    @classmethod
    def _generate_all(
        cls, num_samples: int, n_pairs: int, seed: int, neg_fraction: float
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        gen = random.Random(seed)
        seqs: List[torch.Tensor] = []
        labels: List[int] = []
        for _ in range(num_samples):
            seq = cls._gen_balanced(gen, n_pairs)
            label = 1
            # corrupt to create a negative example
            if gen.random() < neg_fraction:
                idx = gen.randrange(len(seq))
                seq[idx] = cls.CLOSE if seq[idx] == cls.OPEN else cls.OPEN
                label = 0
            seqs.append(torch.tensor(seq, dtype=torch.long))
            labels.append(label)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return seqs, labels_tensor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self._seqs[idx], self._labels[idx]

    @classmethod
    def splits(
        cls,
        num_samples: int,
        n_pairs: int,
        *,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 0,
        neg_fraction: float = 0.5,
    ) -> Tuple["DyckDataset", "DyckDataset", "DyckDataset"]:
        n_train, n_val, _ = _split_sizes(num_samples, train_frac, val_frac)
        train = cls(
            num_samples,
            n_pairs,
            seed=seed,
            neg_fraction=neg_fraction,
            start_idx=0,
            end_idx=n_train,
        )
        val = cls(
            num_samples,
            n_pairs,
            seed=seed,
            neg_fraction=neg_fraction,
            start_idx=n_train,
            end_idx=n_train + n_val,
        )
        test = cls(
            num_samples,
            n_pairs,
            seed=seed,
            neg_fraction=neg_fraction,
            start_idx=n_train + n_val,
        )
        return train, val, test


################################################################################
# 4. Majority‑HeadPad (all signal in head; tail is PAD)
################################################################################


class MajorityHeadPadDataset(Dataset):
    """Majority task where all non‑zero events are restricted to the head.

    - Sequence length = ``length``.
    - The final ``tail_pad_len`` steps are PAD=0.
    - The first ``length - tail_pad_len`` steps contain ±1 events drawn like MajorityDataset.
    - Label is the majority over the entire sequence (equivalently the head only).

    This construction makes the last ``tail_pad_len`` steps uninformative. If you
    train with chunking at ``chunk_size == tail_pad_len`` and cut RTU traces at the
    last chunk boundary, local per‑timestep grads at the last chunk will not be
    able to update the input map or dynamics parameters (no x_t signal), so training
    relies on cross‑boundary corrections. Disabling traces should collapse accuracy
    toward chance.
    """

    PAD, POS, NEG = 0, 1, 2

    def __init__(
        self,
        num_samples: int,
        length: int,
        tail_pad_len: int,
        *,
        nonzero_prob: float = 0.3,
        seed: int = 0,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert 0 < tail_pad_len < length, "tail_pad_len must be in (0, length)"
        self.num_samples, self.length = num_samples, length
        self.tail_pad_len = tail_pad_len
        self.nonzero_prob = nonzero_prob
        self.seed = seed

        seqs, labels = self._generate_all(num_samples, length, tail_pad_len, nonzero_prob, seed)
        end_idx = num_samples if end_idx is None else end_idx
        self._seqs = seqs[start_idx:end_idx]
        self._labels = labels[start_idx:end_idx]
        self.ids = list(range(start_idx, end_idx))

    @staticmethod
    def _generate_all(
        num_samples: int, length: int, tail_pad_len: int, nonzero_prob: float, seed: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        seqs: List[torch.Tensor] = []
        labels: List[int] = []
        head_len = length - tail_pad_len
        gen = torch.Generator().manual_seed(seed)
        for _ in range(num_samples):
            # Head events
            mask_head = torch.rand(head_len, generator=gen) < nonzero_prob
            signs_head = torch.where(
                torch.rand(head_len, generator=gen) < 0.5,
                MajorityHeadPadDataset.NEG,
                MajorityHeadPadDataset.POS,
            )
            head = torch.where(mask_head, signs_head, MajorityHeadPadDataset.PAD)
            tail = torch.zeros(tail_pad_len, dtype=torch.long)
            seq = torch.cat([head, tail], dim=0)
            pos_cnt = (seq == MajorityHeadPadDataset.POS).sum().item()
            neg_cnt = (seq == MajorityHeadPadDataset.NEG).sum().item()
            label = int(pos_cnt > neg_cnt)
            seqs.append(seq.long())
            labels.append(label)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return seqs, labels_tensor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self._seqs[idx], self._labels[idx]

    @classmethod
    def splits(
        cls,
        num_samples: int,
        length: int,
        tail_pad_len: int,
        *,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        nonzero_prob: float = 0.3,
        seed: int = 0,
    ) -> Tuple["MajorityHeadPadDataset", "MajorityHeadPadDataset", "MajorityHeadPadDataset"]:
        n_train, n_val, _ = _split_sizes(num_samples, train_frac, val_frac)
        train = cls(
            num_samples,
            length,
            tail_pad_len,
            nonzero_prob=nonzero_prob,
            seed=seed,
            start_idx=0,
            end_idx=n_train,
        )
        val = cls(
            num_samples,
            length,
            tail_pad_len,
            nonzero_prob=nonzero_prob,
            seed=seed,
            start_idx=n_train,
            end_idx=n_train + n_val,
        )
        test = cls(
            num_samples,
            length,
            tail_pad_len,
            nonzero_prob=nonzero_prob,
            seed=seed,
            start_idx=n_train + n_val,
        )
        return train, val, test
