"""Shared memory record structures for transformer-style policies."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import Tensor


@dataclass(slots=True)
class SegmentMemoryRecord:
    """Snapshot of policy memory captured at the start of a replay segment."""

    segment_index: int
    memory: Optional[Dict[str, Optional[List[Tensor]]]]
