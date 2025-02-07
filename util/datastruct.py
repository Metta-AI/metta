from omegaconf import DictConfig


def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flatten a nested dictionary using dot notation.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key (used in recursion).
        sep (str): Separator between keys.

    Returns:
        dict: A flattened dictionary with keys in dot notation.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.update(flatten_dict(dict(v), new_key, sep=sep))
            continue
        else:
            items[new_key] = v
    return items
