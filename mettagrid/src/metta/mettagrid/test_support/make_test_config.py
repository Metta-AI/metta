"""Configuration utilities for testing."""

def make_test_config(**kwargs) -> dict:
    """Create a test configuration dictionary with default values.
    
    Args:
        **kwargs: Override any default configuration values
        
    Returns:
        dict: Test environment configuration dictionary for use with from_mettagrid_config
    """
    defaults = {
        "max_steps": 100,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
        "inventory_item_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            "move_8way": {"enabled": False},
        },
        "groups": {"player": {"id": 0, "props": {}}},
        "objects": {"wall": {"type_id": 1}},
        "agent": {},
    }
    
    # Deep merge for nested dicts like actions
    if "actions" in kwargs:
        defaults["actions"] = {**defaults["actions"], **kwargs.pop("actions")}
    
    # Merge remaining kwargs with defaults
    config_dict = {**defaults, **kwargs}
    
    return config_dict