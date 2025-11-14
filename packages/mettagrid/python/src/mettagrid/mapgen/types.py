from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# We store maps as 2D arrays of object names.
# "empty" means an empty cell; "wall" means a wall, etc.
MapGrid: TypeAlias = npt.NDArray[np.str_]
# Allow sufficiently descriptive map names (e.g., clipped extractor variants).
map_grid_dtype = np.dtype("<U64")
