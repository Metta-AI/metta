from __future__ import annotations

import uuid

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.s3_policy_spec_loader import policy_spec_from_s3_submission
from metta.tools.utils.auto_config import auto_stats_server_uri
from mettagrid.util.url_schemes import ParsedScheme, SchemeResolver, resolve_uri


class MettaSchemeResolver(SchemeResolver):
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

    def resolve(self, uri: str) -> str:
        parsed = self.parse(uri)
        path = parsed.path
        if not path:
            raise ValueError(f"Invalid metta:// URI: {uri}")

        path_parts = path.split("/")
        if len(path_parts) < 2 or path_parts[0] != "policy":
            raise ValueError(f"Unsupported metta:// URI format: {uri}. Expected metta://policy/<policy_version_id>")

        policy_version_id_str = path_parts[1]
        try:
            policy_version_id = uuid.UUID(policy_version_id_str)
        except ValueError as e:
            raise ValueError(f"Invalid policy version ID in URI: {policy_version_id_str}") from e

        stats_server_uri = auto_stats_server_uri()
        if not stats_server_uri:
            raise ValueError("Cannot resolve metta:// URI: stats server not configured")

        stats_client = StatsClient.create(stats_server_uri)
        policy_version = stats_client.get_policy_version(policy_version_id)

        checkpoint_uri: str | None = None
        if policy_version.s3_path:
            with policy_spec_from_s3_submission(policy_version.s3_path) as policy_spec:
                checkpoint_uri = policy_spec.init_kwargs.get("checkpoint_uri")

        # This is for backwards compatibility; we should remove it when
        # we remove the policy_spec column from the policy_version table.
        if not checkpoint_uri and policy_version.policy_spec:
            checkpoint_uri = policy_version.policy_spec.get("init_kwargs", {}).get("checkpoint_uri")

        if not checkpoint_uri:
            raise ValueError(f"Policy version {policy_version_id} has no checkpoint_uri in policy_spec or s3_path")

        return resolve_uri(checkpoint_uri)
