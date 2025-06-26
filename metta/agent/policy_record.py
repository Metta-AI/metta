"""Compatibility shim - PolicyRecord is now in policy_store.py."""

from metta.agent.policy_store import PolicyRecord

__all__ = ["PolicyRecord"]
