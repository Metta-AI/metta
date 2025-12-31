"""URI scheme tests for checkpoint loading."""

import pytest

from metta.rl.metta_scheme_resolver import MettaSchemeResolver
from mettagrid.util.uri_resolvers.schemes import parse_uri


class TestS3URIs:
    def test_checkpoint_info_parsing(self):
        info = parse_uri("s3://bucket/foo/checkpoints/foo:v9.mpt", allow_none=False).checkpoint_info
        assert info is not None
        run_name, epoch = info
        assert run_name == "foo"
        assert epoch == 9


class TestMettaURIs:
    def test_parsed_uri_parses_metta_scheme(self):
        parsed = parse_uri("metta://policy/acee831a-f409-4345-9c44-79b34af17c3e", allow_none=False)
        assert parsed.scheme == "metta"
        assert parsed.path == "policy/acee831a-f409-4345-9c44-79b34af17c3e"

    def test_resolve_metta_uri_invalid_format(self):
        resolver = MettaSchemeResolver()
        with pytest.raises(ValueError, match="Unsupported metta:// URI format"):
            resolver.get_path_to_policy_spec_or_mpt("metta://invalid")
