from typing import Optional

import torch
from tensordict import TensorClass


class PolicyState(TensorClass):
    lstm_h: Optional[torch.Tensor] = None
    lstm_c: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None
    # Exponential memory traces
    memory_traces: Optional[torch.Tensor] = None  # Shape: (num_agents, num_traces, trace_dim)
    trace_weights: Optional[torch.Tensor] = None  # Shape: (num_agents, num_traces, trace_dim, trace_dim)
