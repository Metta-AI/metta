from pathlib import Path
from typing import Optional

from metta.agent.policy import PolicySpec


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.
    Args:
        policy: Either a shorthand like "random", "simple", "lstm"
                or a full class path like "cogames.policy.random.RandomPolicy"
    Returns:
        Full class path to the policy
    """
    return {
        "random": "cogames.policy.random.RandomPolicy",
        "simple": "cogames.policy.simple.SimplePolicy",
        "lstm": "cogames.policy.lstm.LSTMPolicy",
        "claude": "cogames.policy.claude.ClaudePolicy",
    }.get(policy, policy)


def resolve_policy_data_path(policy_data_path: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint path if provided."""
    if policy_data_path is None:
        return None
    path = Path(policy_data_path)
    if path.is_file():
        return str(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    last_touched_checkpoint_file = max(
        (p for p in path.rglob("*.pt")), key=lambda target: target.stat().st_mtime, default=None
    )
    if not last_touched_checkpoint_file:
        raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
    return str(last_touched_checkpoint_file)


def parse_policy_spec(spec: str) -> PolicySpec:
    """Parse a policy CLI option into its components.

    Args:
        spec: string in the form "policy_class:[proportion][:policy_data]".

    policy_class is a shorthand or full class path to the policy.
    proportion is an optional non-negative float. If omitted, defaults to 1.
    policy_data is an optional path to a policy data file or directory containing policy data files.

    Returns:
        A list of PolicySpec objects

    Raises:
        typer.BadParameter: If the specification is malformed or invalid.
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("Policy specification cannot be empty.")

    parts = raw.split(":", maxsplit=2)
    if len(parts) < 2:
        raise ValueError("Policy specification must include both class path and proportion separated by ':'")
    elif len(parts) > 3:
        raise ValueError("Policy specification must include at most two ':' separated values.")

    raw_class_path, raw_fraction = parts[0].strip(), parts[1].strip()
    raw_policy_data = parts[2].strip() if len(parts) == 3 else None

    if not raw_class_path:
        raise ValueError("Policy class path cannot be empty.")

    if not raw_fraction:
        fraction = 1.0
    else:
        try:
            fraction = float(raw_fraction)
        except ValueError as exc:
            raise ValueError(f"Invalid proportion value '{raw_fraction}'.") from exc

        if fraction <= 0:
            raise ValueError("Policy proportion must be a positive number.")

    resolved_class_path = resolve_policy_class_path(raw_class_path)
    resolved_policy_data = resolve_policy_data_path(raw_policy_data)

    return PolicySpec(
        policy_class_path=resolved_class_path,
        proportion=fraction,
        policy_data_path=resolved_policy_data,
    )
