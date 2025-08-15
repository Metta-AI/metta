import numpy as np
from typing_extensions import Any


def clean_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types, preserving integer types."""
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            item = obj.item()
            # Preserve integer types
            if isinstance(item, (np.integer, int)) and not isinstance(item, bool):
                return int(item)
            elif isinstance(item, (np.floating, float)):
                return float(item)
            else:
                return item
        else:
            return obj.tolist()
    elif isinstance(obj, np.generic):
        item = obj.item()
        # Preserve integer types
        if isinstance(item, (np.integer, int)) and not isinstance(item, bool):
            return int(item)
        elif isinstance(item, (np.floating, float)):
            return float(item)
        else:
            return item
    elif isinstance(obj, dict):
        return {k: clean_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(v) for v in obj]
    return obj
