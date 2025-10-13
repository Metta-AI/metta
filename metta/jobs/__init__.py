"""Shared job infrastructure for Metta systems.

This package provides common components used by:
- Experiment system (metta/experiment/)
- Stable release system (devops/stable/)
- Adaptive sweeps (metta/adaptive/)
"""

from metta.jobs.models import JobSpec

__all__ = ["JobSpec"]
