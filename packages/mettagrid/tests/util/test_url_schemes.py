import pytest

from mettagrid.util.uri_resolvers.schemes import (
    FileSchemeResolver,
    MockSchemeResolver,
    S3SchemeResolver,
    parse_uri,
    resolve_uri,
)


class TestFileSchemeResolver:
    def test_parse_file_uri(self, tmp_path):
        resolver = FileSchemeResolver()
        parsed = resolver.parse(f"file://{tmp_path}/test.txt")
        assert parsed.scheme == "file"
        assert parsed.local_path is not None
        assert str(parsed.local_path).endswith("test.txt")

    def test_parse_plain_path(self, tmp_path):
        resolver = FileSchemeResolver()
        parsed = resolver.parse(str(tmp_path / "test.txt"))
        assert parsed.scheme == "file"
        assert parsed.local_path is not None


class TestS3SchemeResolver:
    def test_parse_s3_uri(self):
        resolver = S3SchemeResolver()
        parsed = resolver.parse("s3://bucket/path/to/file.txt")
        assert parsed.scheme == "s3"
        assert parsed.bucket == "bucket"
        assert parsed.key == "path/to/file.txt"

    def test_parse_s3_uri_missing_key(self):
        resolver = S3SchemeResolver()
        with pytest.raises(ValueError, match="Malformed S3 URI"):
            resolver.parse("s3://bucket")


class TestMockSchemeResolver:
    def test_parse_mock_uri(self):
        resolver = MockSchemeResolver()
        parsed = resolver.parse("mock://test_policy")
        assert parsed.scheme == "mock"
        assert parsed.path == "test_policy"

    def test_parse_mock_uri_empty_path(self):
        resolver = MockSchemeResolver()
        with pytest.raises(ValueError, match="must include a path"):
            resolver.parse("mock://")


class TestParseUri:
    def test_parse_file_uri(self, tmp_path):
        parsed = parse_uri(f"file://{tmp_path}/test.txt")
        assert parsed.scheme == "file"

    def test_parse_s3_uri(self):
        parsed = parse_uri("s3://bucket/key")
        assert parsed.scheme == "s3"

    def test_parse_mock_uri(self):
        parsed = parse_uri("mock://policy")
        assert parsed.scheme == "mock"

    def test_parse_metta_uri(self):
        parsed = parse_uri("metta://policy/123")
        assert parsed.scheme == "metta"

    def test_parse_plain_path(self, tmp_path):
        parsed = parse_uri(str(tmp_path / "test.txt"))
        assert parsed.scheme == "file"

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid URI"):
            parse_uri("unknown://path")


class TestResolveUri:
    def test_resolve_file_uri(self, tmp_path):
        uri = f"file://{tmp_path}/test.txt"
        parsed = resolve_uri(uri)
        assert parsed.canonical.startswith("file://")

    def test_resolve_plain_path(self, tmp_path):
        path = str(tmp_path / "test.txt")
        parsed = resolve_uri(path)
        assert parsed.canonical.startswith("file://")
