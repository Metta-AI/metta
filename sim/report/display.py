"""
Utility functions for display names and formatting.
"""

from typing import Dict, Optional

def get_display_name(policy_name: str, policy_names: Optional[Dict[str, str]] = None) -> str:
    if policy_names is not None and policy_name in policy_names:
        return policy_names[policy_name]
    return policy_name

def get_short_eval_name(eval_name: str) -> str:
    return eval_name.split('/')[-1]

def format_metric_name(metric: str) -> str:
    # Replace underscores with spaces and capitalize words
    return ' '.join(word.capitalize() for word in metric.replace('.', '_').split('_'))