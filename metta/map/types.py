from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Shaped version, `np.ndarray[tuple[int, int], np.dtype[np.str_]]`,
# would be better, but slices from numpy arrays are not typed properly, which makes it too annoying to use.
MapGrid: TypeAlias = npt.NDArray[np.str_]
