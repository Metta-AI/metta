from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cogames.aws_storage import DownloadOutcome, maybe_download_checkpoint
from cogames.policy.policy import PolicySpec

if TYPE_CHECKING:
    from rich.console import Console


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path."""
    return {
        "random": "cogames.policy.random.RandomPolicy",
        "simple": "cogames.policy.simple.SimplePolicy",
        "lstm": "cogames.policy.lstm.LSTMPolicy",
        "claude": "cogames.policy.claude.ClaudePolicy",
    }.get(policy, policy)


def resolve_policy_data_path(
    policy_data_path: Optional[str],
    *,
    policy_class_path: Optional[str] = None,
    game_name: Optional[str] = None,
    console: Optional["Console"] = None,
) -> Optional[str]:
    """Resolve the checkpoint path, downloading from S3 when configured."""

    if policy_data_path is None:
        return None

    path = Path(policy_data_path).expanduser()
    if path.is_file():
        return str(path)

    if path.is_dir():
        latest_checkpoint = max(
            (candidate for candidate in path.rglob("*.pt")),
            key=lambda candidate: candidate.stat().st_mtime,
            default=None,
        )
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
        return str(latest_checkpoint)

    if path.exists():
        return str(path)

    if console is not None and policy_class_path is not None and path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome: DownloadOutcome = maybe_download_checkpoint(
            policy_path=path,
            game_name=game_name,
            policy_class_path=policy_class_path,
            console=console,
        )
        if outcome.downloaded:
            return str(path)

    # If we reach here, the path is still unavailable.
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def parse_policy_spec(
    spec: str,
    *,
    console: Optional["Console"] = None,
    game_name: Optional[str] = None,
) -> PolicySpec:
    """Parse a CLI policy specification string."""
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    parts = [part.strip() for part in raw.split(":")]
    if len(parts) > 3:
        raise ValueError("Policy specification must include at most two ':' separators.")

    raw_class_path = parts[0]
    raw_policy_data = parts[1] if len(parts) > 1 else None
    raw_proportion = parts[2] if len(parts) > 2 else None

    if not raw_class_path:
        raise ValueError("Policy class path cannot be empty.")

    if raw_proportion is None:
        proportion = 1.0
    else:
        try:
            proportion = float(raw_proportion)
        except ValueError as exc:  # pragma: no cover - user input error path
            raise ValueError(f"Invalid proportion value '{raw_proportion}'.") from exc
        if proportion <= 0:
            raise ValueError("Policy proportion must be a positive number.")

    resolved_class_path = resolve_policy_class_path(raw_class_path)
    resolved_policy_data = resolve_policy_data_path(
        raw_policy_data,
        policy_class_path=resolved_class_path,
        game_name=game_name,
        console=console,
    )

    return PolicySpec(
        policy_class_path=resolved_class_path,
        policy_data_path=resolved_policy_data,
        proportion=proportion,
    )
