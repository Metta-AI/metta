import numpy as np
from numpy.typing import NDArray

def compute_gae(
    dones: NDArray[np.float32],
    values: NDArray[np.float32],
    rewards: NDArray[np.float32],
    gamma: float,
    gae_lambda: float,
) -> NDArray[np.float32]: ...
