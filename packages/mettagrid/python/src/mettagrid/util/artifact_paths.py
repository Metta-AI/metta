"""Helpers for constructing artifact paths (local or remote)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

from mettagrid.util.file import http_url
from mettagrid.util.uri import ParsedURI

ArtifactBase = Union[str, Path]


def _clean_segments(segments: Iterable[str]) -> list[str]:
    return [seg.strip("/") for seg in segments if seg and seg.strip("/")]


def _join_value(base: ArtifactBase, cleaned: list[str]) -> ArtifactBase:
    if not cleaned:
        return base

    if isinstance(base, Path):
        return base.joinpath(*cleaned)

    base_str = str(base)

    try:
        parsed = ParsedURI.parse(base_str)
    except ValueError:
        parsed = None

    if parsed is not None:
        if parsed.scheme == "file" and parsed.local_path is not None:
            result_path = parsed.local_path.joinpath(*cleaned)
            return str(result_path)

        if parsed.scheme == "s3":
            key_parts = []
            if parsed.key:
                key_parts.append(parsed.key.rstrip("/"))
            key_parts.extend(cleaned)
            key = "/".join(part for part in key_parts if part)
            if parsed.bucket is None:
                return f"s3:///{key}" if key else "s3:///"
            return f"s3://{parsed.bucket}/{key}" if key else f"s3://{parsed.bucket}"

        if parsed.scheme == "gdrive":
            raw_path = parsed.path or ""
            if raw_path.startswith("gdrive://"):
                raw_path = raw_path[len("gdrive://") :]
            prefix = raw_path.rstrip("/")
            joined = "/".join(filter(None, [prefix, *cleaned]))
            return f"gdrive://{joined}" if joined else parsed.raw

        if parsed.scheme == "mock":
            path = "/".join(filter(None, [(parsed.path or "").rstrip("/"), *cleaned]))
            return f"mock://{path}"

        if parsed.scheme in {"http", "https"}:
            base_http = base_str.rstrip("/")
            suffix = "/".join(cleaned)
            return f"{base_http}/{suffix}"

    result_path = Path(base_str).joinpath(*cleaned)
    return str(result_path)


@dataclass(frozen=True)
class ArtifactReference:
    """Normalized wrapper around an artifact root (local path or URI)."""

    value: ArtifactBase

    def __post_init__(self) -> None:  # pragma: no cover - simple normalization
        if isinstance(self.value, str):
            normalized = self.value.strip()
            if not normalized:
                raise ValueError("ArtifactReference cannot wrap an empty string")
            object.__setattr__(self, "value", normalized)

    def join(self, *segments: str) -> "ArtifactReference":
        return ArtifactReference(_join_value(self.value, _clean_segments(segments)))

    def as_str(self) -> str:
        return str(self.value) if not isinstance(self.value, Path) else str(self.value)

    def __str__(self) -> str:  # pragma: no cover - convenience repr
        return self.as_str()

    def with_policy(self, run_name: Optional[str], epoch: Optional[int]) -> "ArtifactReference":
        """Return a replay root nested under a policy run/epoch."""

        if not run_name:
            return self
        policy_root = self.join(run_name)
        if epoch:
            return policy_root.join(f"v{epoch}")
        return policy_root

    def with_simulation(
        self,
        suite: str,
        name: str,
        *,
        simulation_id: Optional[str] = None,
    ) -> "ArtifactReference":
        """Append simulation suite/name (and optional ID) to this reference."""

        sim_root = self.join(suite, name)
        if simulation_id:
            return sim_root.join(simulation_id)
        return sim_root

    def as_path(self) -> Path:
        if isinstance(self.value, Path):
            return self.value
        parsed = ParsedURI.parse(str(self.value))
        if parsed.scheme == "file" and parsed.local_path is not None:
            return parsed.local_path
        raise ValueError(f"Artifact '{self.value}' does not reference a local path")

    def to_public_url(self) -> Optional[str]:
        candidate = http_url(str(self.value))
        if candidate.startswith("file://"):
            return None
        if candidate == str(self.value) and not candidate.startswith(("http://", "https://")):
            return None
        return candidate

    def is_remote(self) -> bool:
        if isinstance(self.value, Path):
            return False
        parsed = ParsedURI.parse(str(self.value))
        return parsed.is_remote()


def ensure_artifact_reference(value: ArtifactBase | ArtifactReference | None) -> Optional[ArtifactReference]:
    if value is None:
        return None
    if isinstance(value, ArtifactReference):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("Artifact value cannot be empty")
        return ArtifactReference(normalized)
    return ArtifactReference(value)


def artifact_policy_run_root(
    base: ArtifactBase | ArtifactReference | None,
    *,
    run_name: str | None,
    epoch: int | None,
) -> Optional[ArtifactReference]:
    """Return the replay root for a policy run under *base*.

    ``base`` can be a filesystem path or URI prefix. ``run_name`` is required for
    stable directory layouts; if it is falsy the base is returned verbatim. When
    ``epoch`` is truthy, a ``v{epoch}`` component is appended to produce versioned
    replay buckets.
    """

    ref = ensure_artifact_reference(base)
    if ref is None or not run_name:
        return ref

    return ref.with_policy(run_name, epoch)


def artifact_simulation_root(
    base: ArtifactBase | ArtifactReference | None,
    *,
    suite: str,
    name: str,
    simulation_id: str | None = None,
) -> Optional[ArtifactReference]:
    """Build the replay directory root for a simulation."""

    ref = ensure_artifact_reference(base)
    if ref is None:
        return None
    return ref.with_simulation(suite, name, simulation_id=simulation_id)


@dataclass(frozen=True)
class PolicyArtifactLayout:
    """Precomputed layout for policy checkpoints and replays."""

    run_name: Optional[str]
    checkpoint_root: Optional[ArtifactReference] = None
    replay_root: Optional[ArtifactReference] = None

    @classmethod
    def build(
        cls,
        *,
        run_name: Optional[str],
        checkpoint_base: ArtifactBase | ArtifactReference | None = None,
        checkpoint_includes_run: bool = False,
        replay_base: ArtifactBase | ArtifactReference | None = None,
        epoch: Optional[int] = None,
    ) -> "PolicyArtifactLayout":
        checkpoint_root: Optional[ArtifactReference] = None
        checkpoint_ref = ensure_artifact_reference(checkpoint_base)
        if checkpoint_ref is not None:
            if run_name and not checkpoint_includes_run:
                checkpoint_root = checkpoint_ref.join(run_name)
            else:
                checkpoint_root = checkpoint_ref

        replay_root: Optional[ArtifactReference] = None
        replay_ref = ensure_artifact_reference(replay_base)
        if replay_ref is not None:
            if run_name:
                replay_root = replay_ref.with_policy(run_name, epoch)
            elif epoch:
                replay_root = replay_ref.join(f"v{epoch}")
            else:
                replay_root = replay_ref

        return cls(run_name=run_name, checkpoint_root=checkpoint_root, replay_root=replay_root)

    def checkpoints_dir(self) -> Optional[ArtifactReference]:
        if self.checkpoint_root is None:
            return None
        return self.checkpoint_root.join("checkpoints")

    def checkpoint_file(self, filename: str) -> Optional[ArtifactReference]:
        checkpoints = self.checkpoints_dir()
        if checkpoints is None:
            return None
        return checkpoints.join(filename)

    def simulation_root(
        self,
        suite: str,
        name: str,
        *,
        simulation_id: Optional[str] = None,
    ) -> Optional[ArtifactReference]:
        if self.replay_root is None:
            return None
        return self.replay_root.with_simulation(suite, name, simulation_id=simulation_id)

    def simulation_roots(self, simulations: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], ArtifactReference]:
        roots: Dict[Tuple[str, str], ArtifactReference] = {}
        if self.replay_root is None:
            return roots
        for suite, name in simulations:
            roots[(suite, name)] = self.replay_root.with_simulation(suite, name)
        return roots


class ArtifactRef(str):
    """Pydantic-friendly wrapper around canonical artifact strings."""

    def __new__(cls, value: str | Path | ArtifactReference | "ArtifactRef") -> "ArtifactRef":  # type: ignore[override]
        if isinstance(value, ArtifactRef):
            return str.__new__(cls, str(value))
        ref = ensure_artifact_reference(value)
        if ref is None:
            raise ValueError("Artifact value cannot be None")
        return str.__new__(cls, ref.as_str())

    def as_reference(self) -> ArtifactReference:
        return ArtifactReference(str(self))

    def join(self, *segments: str) -> "ArtifactRef":
        return ArtifactRef(self.as_reference().join(*segments).as_str())

    @classmethod
    def __get_validators__(cls):  # pragma: no cover - exercised via Pydantic
        yield cls.validate

    @classmethod
    def validate(cls, value: str | Path | ArtifactReference | "ArtifactRef") -> "ArtifactRef":
        return cls(value)


__all__ = [
    "ArtifactReference",
    "ArtifactRef",
    "PolicyArtifactLayout",
    "artifact_policy_run_root",
    "artifact_simulation_root",
    "ensure_artifact_reference",
]
