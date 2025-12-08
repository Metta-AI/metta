"""
Utility functions for the Policy Selector Widget
"""

from typing import Any, Dict, List

from metta.app_backend.clients.scorecard_client import ScorecardClient


async def fetch_policies_from_backend(
    base_url: str = "http://localhost:8000",
) -> List[Dict[str, Any]]:
    """Fetch policies from the backend API.

    Args:
        base_url: Base URL of the backend API

    Returns:
        List of policy metadata dictionaries
    """
    client = ScorecardClient(base_url=base_url)
    try:
        response = await client.get_policy_versions()
        policies = (
            getattr(response, "entries", None)
            or getattr(response, "policies", [])
            or []
        )

        def _get_attr(policy, key: str):
            if isinstance(policy, dict):
                return policy.get(key)
            return getattr(policy, key, None)

        normalized = []
        for policy in policies:
            tags = _get_attr(policy, "tags") or {}
            normalized.append(
                {
                    "id": _get_attr(policy, "id"),
                    "type": _get_attr(policy, "type")
                    or tags.get("type")
                    or tags.get("policy_type")
                    or "training_run",
                    "name": _get_attr(policy, "name"),
                    "user_id": _get_attr(policy, "user_id"),
                    "created_at": _get_attr(policy, "created_at"),
                    "tags": tags,
                }
            )

        return normalized
    except Exception as e:
        print(f"Error fetching policies: {e}")
        return []


def filter_policies(
    policies: List[Dict[str, Any]],
    search_term: str = "",
    policy_types: List[str] = None,
    tags: List[str] = None,
) -> List[Dict[str, Any]]:
    """Filter policies based on search criteria.

    Args:
        policies: List of policy metadata dicts
        search_term: Text to search for in policy names
        policy_types: List of policy types to include
        tags: List of tags that policies must have

    Returns:
        Filtered list of policies
    """
    filtered = policies

    # Filter by search term
    if search_term:
        search_lower = search_term.lower()
        filtered = [p for p in filtered if search_lower in p.get("name", "").lower()]

    # Filter by policy types
    if policy_types:
        filtered = [p for p in filtered if p.get("type") in policy_types]

    # Filter by tags
    if tags:
        filtered = [
            p for p in filtered if any(tag in p.get("tags", []) for tag in tags)
        ]

    return filtered


def group_policies_by_type(
    policies: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group policies by their type.

    Args:
        policies: List of policy metadata dicts

    Returns:
        Dictionary mapping policy types to lists of policies
    """
    groups = {}
    for policy in policies:
        policy_type = policy.get("type", "unknown")
        if policy_type not in groups:
            groups[policy_type] = []
        groups[policy_type].append(policy)

    return groups


def sort_policies(
    policies: List[Dict[str, Any]], sort_by: str = "name", reverse: bool = False
) -> List[Dict[str, Any]]:
    """Sort policies by a given field.

    Args:
        policies: List of policy metadata dicts
        sort_by: Field to sort by ("name", "created_at", "type")
        reverse: Whether to sort in reverse order

    Returns:
        Sorted list of policies
    """
    if sort_by == "created_at":
        return sorted(policies, key=lambda p: p.get("created_at", ""), reverse=reverse)
    elif sort_by == "type":
        return sorted(policies, key=lambda p: p.get("type", ""), reverse=reverse)
    else:  # default to name
        return sorted(policies, key=lambda p: p.get("name", ""), reverse=reverse)
