from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import PolicyVersionWithName
from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.rl.system_config import guess_data_dir
from mettagrid.util.uri_resolvers.base import MettaParsedScheme, SchemeResolver
from mettagrid.util.uri_resolvers.schemes import resolve_uri

logger = logging.getLogger(__name__)


def _is_uuid(s: str) -> bool:
    """Check if string is a valid UUID.

    Returns True if valid UUID, False if clearly not a UUID.
    Raises ValueError if the string looks like a UUID (correct format) but is invalid.
    """
    if len(s) != 36:
        return False
    if not (s[8] == "-" and s[13] == "-" and s[18] == "-" and s[23] == "-"):
        return False

    try:
        uuid.UUID(s)
        return True
    except ValueError as e:
        raise ValueError(f"Invalid policy version ID: {s}") from e


def _parse_policy_identifier(path: str) -> tuple[str, int | None]:
    """Parse a policy identifier into (name, version).

    Supports:
    - policy-name -> (policy-name, None) meaning latest
    - policy-name:latest -> (policy-name, None) meaning latest
    - policy-name:vX -> (policy-name, X) meaning specific version
    """
    if path.endswith(":latest"):
        return path[:-7], None
    if ":v" in path:
        run_name, suffix = path.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return path, None


class MettaSchemeResolver(SchemeResolver):
    """Resolves metta:// URIs to checkpoint URIs via the stats server.

    Supported formats:
      - metta://policy/<uuid>              (policy version by UUID)
      - metta://policy/<name>              (latest version of named policy)
      - metta://policy/<name>:latest       (latest version of named policy)
      - metta://policy/<name>:v<N>         (specific version N of named policy)
    """

    def __init__(self, stats_server_uri: str | None = None):
        self._stats_server_uri = stats_server_uri or os.environ.get("STATS_SERVER_URI") or PROD_STATS_SERVER_URI

    @property
    def scheme(self) -> str:
        return "metta"

    def parse(self, uri: str) -> MettaParsedScheme:
        if not uri.startswith("metta://"):
            raise ValueError(f"Expected metta:// URI, got: {uri}")
        path = uri[len("metta://") :]
        if not path:
            raise ValueError("metta:// URIs must include a path")
        return MettaParsedScheme(canonical=uri, path=path)

    def get_policy_version(self, uri: str) -> PolicyVersionWithName:
        parsed = self.parse(uri)
        path = parsed.path
        if not path:
            raise ValueError(f"Invalid metta:// URI: {uri}")

        path_parts = path.split("/")
        if len(path_parts) < 2 or path_parts[0] != "policy":
            raise ValueError(
                f"Unsupported metta:// URI format: {uri}. "
                f"Expected metta://policy/<policy_version_id> or metta://policy/<policy_name>"
            )

        if not self._stats_server_uri:
            raise ValueError("Cannot resolve metta:// URI: stats server not configured")
        stats_client = StatsClient.create(self._stats_server_uri)

        if _is_uuid(path_parts[1]):
            policy_version = stats_client.get_policy_version(uuid.UUID(path_parts[1]))
        else:
            name, version = _parse_policy_identifier(path_parts[1])
            response = stats_client.get_policy_versions(name_exact=name, version=version, limit=1)
            if not response.entries:
                version_str = f":v{version}" if version is not None else ""
                raise ValueError(f"No policy found with name '{name}{version_str}'")
            entry = response.entries[0]
            policy_version = stats_client.get_policy_version(entry.id)

        return policy_version

    def get_path_to_policy_spec(self, uri: str) -> str:
        parsed = self.parse(uri)
        path_parts = parsed.path.split("/")
        if len(path_parts) < 2 or path_parts[0] != "policy":
            raise ValueError(
                f"Unsupported metta:// URI format: {uri}. "
                f"Expected metta://policy/<policy_version_id> or metta://policy/<policy_name>"
            )
        if not _is_uuid(path_parts[1]):
            name, version = _parse_policy_identifier(path_parts[1])
            checkpoint_root = Path(guess_data_dir()).expanduser().resolve() / name / "checkpoints"
            if checkpoint_root.exists():
                candidate = checkpoint_root if version is None else checkpoint_root / f"{name}:v{version}"
                if candidate.exists():
                    logger.info("Metta scheme resolver: %s resolved to local checkpoint %s", uri, candidate.as_uri())
                    return candidate.as_uri()

        policy_version = self.get_policy_version(uri)
        if policy_version.s3_path:
            resolved = resolve_uri(policy_version.s3_path).canonical
            logger.info("Metta scheme resolver: %s resolved to s3 policy spec: %s", uri, resolved)
            return resolved

        raise ValueError(
            f"Policy version {policy_version.id} has no s3_path; "
            "expected a policy spec submission in S3. Legacy .mpt files are not supported."
        )
