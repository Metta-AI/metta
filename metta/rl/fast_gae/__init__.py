# metta/rl/fast_gae/__init__.py

# Tell type checkers to ignore this file
# pyright: reportMissingImports=false
# pylint: disable=import-error, wildcard-import
# flake8: noqa
# mypy: ignore-errors

try:
    from .fast_gae import *  # type: ignore
except ImportError:
    # This will happen before the extension is built
    # Define expected exports to help IDE and static analysis tools
    def compute_gae(*args, **kwargs):
        """
        Placeholder for the Cython function that will be available after building.

        This function will be replaced by the actual implementation when the extension is built.
        """
        raise NotImplementedError(
            "The fast_gae extension has not been built yet. Run 'python setup.py build_ext --inplace' to build it."
        )
