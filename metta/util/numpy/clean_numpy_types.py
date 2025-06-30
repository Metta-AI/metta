import numpy as np


def clean_numpy_types(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.item() if obj.size == 1 else obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: clean_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(v) for v in obj]
    return obj
