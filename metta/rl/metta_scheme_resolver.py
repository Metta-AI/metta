from __future__ import annotations

import logging
import uuid

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import PolicyVersionWithName
from metta.common.util.constants import PROD_STATS_SERVER_URI
from mettagrid.util.uri_resolvers.base import MettaParsedScheme, SchemeResolver

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
        self._stats_server_uri = stats_server_uri or PROD_STATS_SERVER_URI

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

    def _get_stats_client(self) -> StatsClient:
        if not self._stats_server_uri:
            raise ValueError("Cannot resolve metta:// URI: stats server not configured")
        return StatsClient.create(self._stats_server_uri)

    def _resolve_policy_version_by_name(
        self, stats_client: StatsClient, name: str, version: int | None
    ) -> PolicyVersionWithName:
        response = stats_client.get_policy_versions(name_exact=name, version=version, limit=1)
        if not response.entries:
            version_str = f":v{version}" if version is not None else ""
            raise ValueError(f"No policy found with name '{name}{version_str}'")
        entry = response.entries[0]
        return stats_client.get_policy_version(entry.id)

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

        policy_identifier = path_parts[1]
        stats_client = self._get_stats_client()

        if _is_uuid(policy_identifier):
            policy_version = stats_client.get_policy_version(uuid.UUID(policy_identifier))
        else:
            name, version = _parse_policy_identifier(policy_identifier)
            policy_version = self._resolve_policy_version_by_name(stats_client, name, version)

        return policy_version

    def get_path_to_policy_spec_or_mpt(self, uri: str) -> str:
        policy_version = self.get_policy_version(uri)
        # By default we send you to the s3 path that contains the policy spec
        if policy_version.s3_path:
            logger.info(f"Metta scheme resolver: {uri} resolved to s3 policy spec: {policy_version.s3_path}")
            return policy_version.s3_path

        raise ValueError(
            f"Policy version {policy_version.id} has no s3_path; expected a policy spec submission zip in S3."
        )
