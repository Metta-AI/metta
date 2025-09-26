"""Compatibility shim for legacy ``metta.common`` imports.

Implementation details and the supported surface are documented in
``docs/metta_common_shim.md``.  Until the workspace completes the namespace
split, this module must keep forwarding consumers into the ``metta-common``
package without breaking existing imports.
"""

# Intentionally empty: the shim relies on Python's namespace package
# behaviour to expose modules from ``metta-common`` while still providing
# local helpers like ``test_support``.
