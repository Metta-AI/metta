from __future__ import annotations

import logging
import uuid
from pathlib import Path

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import PolicyVersionWithName
from metta.common.s3_policy_spec_loader import policy_spec_from_s3_submission
from metta.tools.utils.auto_config import auto_stats_server_uri
from mettagrid.util.uri_resolvers.base import ParsedScheme, SchemeResolver, _extract_run_and_epoch
from mettagrid.util.uri_resolvers.schemes import resolve_uri

logger = logging.getLogger(__name__)


def _looks_like_uuid_format(s: str) -> bool:
    """Check if string has the structural format of a UUID (8-4-4-4-12 pattern)."""
    if len(s) != 36:
        return False
    return s[8] == "-" and s[13] == "-" and s[18] == "-" and s[23] == "-"


def _is_uuid(s: str) -> bool:
    """Check if string is a valid UUID.

    Returns True if valid UUID, False if clearly not a UUID.
    Raises ValueError if the string looks like a UUID (correct format) but is invalid.
    """
    if not _looks_like_uuid_format(s):
        return False

    try:
        uuid.UUID(s)
        return True
    except ValueError as e:
        raise ValueError(f"Invalid policy version ID: {s}") from e


def _parse_policy_identifier(identifier: str) -> tuple[str, int | None]:
    """Parse a policy identifier into (name, version).

    Supports:
    - policy-name -> (policy-name, None) meaning latest
    - policy-name:latest -> (policy-name, None) meaning latest
    - policy-name:vX -> (policy-name, X) meaning specific version
    """
    if identifier.endswith(":latest"):
        return identifier[:-7], None
    info = _extract_run_and_epoch(Path(identifier))
    if info:
        return info
    return identifier, None


class MettaSchemeResolver(SchemeResolver):
    """Resolves metta:// URIs to checkpoint URIs via the stats server.

    Supported formats:
      - metta://policy/<uuid>              (policy version by UUID)
      - metta://policy/<name>              (latest version of named policy)
      - metta://policy/<name>:latest       (latest version of named policy)
      - metta://policy/<name>:v<N>         (specific version N of named policy)
    """

    @property
    def scheme(self) -> str:
        return "metta"

    def parse(self, uri: str) -> ParsedScheme:
        if not uri.startswith("metta://"):
            raise ValueError(f"Expected metta:// URI, got: {uri}")

        path = uri[len("metta://") :]
        if not path:
            raise ValueError(f"Invalid metta:// URI: {uri}")

        return ParsedScheme(raw=uri, scheme="metta", canonical=uri, path=path)

    def _get_stats_client(self) -> StatsClient:
        stats_server_uri = auto_stats_server_uri()
        if not stats_server_uri:
            raise ValueError("Cannot resolve metta:// URI: stats server not configured")
        return StatsClient.create(stats_server_uri)

    def _resolve_policy_version_by_name(
        self, stats_client: StatsClient, name: str, version: int | None
    ) -> PolicyVersionWithName:
        response = stats_client.get_policies(name_exact=name, version=version, limit=1)
        if not response.entries:
            version_str = f":v{version}" if version is not None else ""
            raise ValueError(f"No policy found with name '{name}{version_str}'")
        entry = response.entries[0]
        return stats_client.get_policy_version(entry.id)

    def resolve(self, uri: str) -> str:
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

        policy_identifier = path_parts[1]
        stats_client = self._get_stats_client()

        if _is_uuid(policy_identifier):
            policy_version = stats_client.get_policy_version(uuid.UUID(policy_identifier))
        else:
            name, version = _parse_policy_identifier(policy_identifier)
            policy_version = self._resolve_policy_version_by_name(stats_client, name, version)

        checkpoint_uri: str | None = None
        if policy_version.s3_path:
            with policy_spec_from_s3_submission(policy_version.s3_path) as policy_spec:
                checkpoint_uri = policy_spec.init_kwargs.get("checkpoint_uri")

        # This is for backwards compatibility; we should remove it when
        # we remove the policy_spec column from the policy_version table.
        if not checkpoint_uri and policy_version.policy_spec:
            checkpoint_uri = policy_version.policy_spec.get("init_kwargs", {}).get("checkpoint_uri")

        if not checkpoint_uri:
            raise ValueError(f"Policy version {policy_version.id} has no checkpoint_uri in policy_spec or s3_path")

        logger.info(f"Metta scheme resolver: {uri} resolved to mpt checkpoint: {checkpoint_uri}")
        return resolve_uri(checkpoint_uri)
