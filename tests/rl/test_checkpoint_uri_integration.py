"""URI scheme tests for checkpoint loading."""

import pytest

from metta.rl.metta_scheme_resolver import MettaSchemeResolver
from mettagrid.util.file import ParsedURI
from mettagrid.util.url_schemes import key_and_version


class TestS3URIs:
    def test_key_and_version_parsing(self):
        key, version = key_and_version("s3://bucket/foo/checkpoints/foo:v9.mpt")
        assert key == "foo"
        assert version == 9


class TestMettaURIs:
    def test_parsed_uri_parses_metta_scheme(self):
        parsed = ParsedURI.parse("metta://policy/acee831a-f409-4345-9c44-79b34af17c3e")
        assert parsed.scheme == "metta"
        assert parsed.path == "policy/acee831a-f409-4345-9c44-79b34af17c3e"

    def test_resolve_metta_uri_invalid_format(self):
        resolver = MettaSchemeResolver()
        with pytest.raises(ValueError, match="Unsupported metta:// URI format"):
            resolver.resolve("metta://invalid")

    def test_resolve_metta_uri_requires_stats_server(self, monkeypatch):
        monkeypatch.setattr("metta.rl.metta_scheme_resolver.auto_stats_server_uri", lambda: None)
        resolver = MettaSchemeResolver()
        with pytest.raises(ValueError, match="stats server not configured"):
            resolver.resolve("metta://policy/acee831a-f409-4345-9c44-79b34af17c3e")
