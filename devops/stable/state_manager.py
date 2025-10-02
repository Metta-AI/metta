"""State management for release validation runs.

Handles persisting and loading release state to/from JSON files.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from devops.stable.domain import ReleaseState


class StateManager:
    """Manages release state persistence."""

    def __init__(self, state_dir: Path = Path("devops/stable/state")):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_commit_sha(self) -> Optional[str]:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def create_state(self, version: str, repo_root: str) -> ReleaseState:
        """Create a new release state."""
        return ReleaseState(
            version=version,
            repo_root=repo_root,
            commit_sha=self._get_commit_sha(),
            created_at=datetime.utcnow(),
        )

    def save_state(self, state: ReleaseState) -> Path:
        """Save release state to JSON file."""
        filename = f"{state.version}.json"
        path = self.state_dir / filename

        # Convert to JSON with datetime serialization
        with open(path, "w") as f:
            json.dump(state.model_dump(mode="json"), f, indent=2, default=str)

        return path

    def load_state(self, version: str) -> Optional[ReleaseState]:
        """Load release state from JSON file."""
        filename = f"{version}.json"
        path = self.state_dir / filename

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)
            return ReleaseState.model_validate(data)

    def list_states(self) -> list[str]:
        """List all available release states (versions)."""
        return sorted([p.stem for p in self.state_dir.glob("*.json")], reverse=True)

    def get_latest_state(self) -> Optional[ReleaseState]:
        """Get the most recent release state."""
        states = self.list_states()
        if not states:
            return None
        return self.load_state(states[0])
