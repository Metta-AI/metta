from pathlib import Path

import pytest

from mettagrid.util.artifact_paths import (
    ArtifactReference,
    artifact_policy_run_root,
    artifact_simulation_root,
    ensure_artifact_reference,
)
from mettagrid.util.uri import ParsedURI


def _join(base, *segments) -> ArtifactReference:
    ref = ensure_artifact_reference(base)
    assert ref is not None
    return ref.join(*segments)


def test_reference_join_s3():
    joined = _join("s3://bucket/replays", "run_a", "sim", "episode.json.z")
    assert joined.as_str() == "s3://bucket/replays/run_a/sim/episode.json.z"


def test_reference_join_s3_with_trailing_slash():
    joined = _join("s3://bucket/replays/", "run_a")
    assert joined.as_str() == "s3://bucket/replays/run_a"


def test_reference_join_s3_bucket_root():
    joined = _join("s3://bucket", "run_a", "episode.json.z")
    assert joined.as_str() == "s3://bucket/run_a/episode.json.z"


def test_reference_join_s3_bucket_root_trailing_slash():
    joined = _join("s3://bucket/", "run_a")
    assert joined.as_str() == "s3://bucket/run_a"


def test_reference_join_local_str(tmp_path: Path):
    joined = _join(str(tmp_path / "replays"), "run_a", "sim")
    assert joined.as_path() == tmp_path / "replays" / "run_a" / "sim"


def test_reference_join_path(tmp_path: Path):
    joined = _join(tmp_path / "policies", "run_a", "checkpoints")
    assert joined.as_path() == tmp_path / "policies" / "run_a" / "checkpoints"


def test_reference_join_gdrive():
    joined = _join("gdrive://folder", "run_a", "episode.json.z")
    assert joined.as_str() == "gdrive://folder/run_a/episode.json.z"


def test_artifact_policy_run_root_s3_epoch():
    root = artifact_policy_run_root("s3://bucket/replays", run_name="run_a", epoch=3)
    assert isinstance(root, ArtifactReference)
    assert str(root.value) == "s3://bucket/replays/run_a/v3"


def test_artifact_policy_run_root_s3_bucket_root():
    root = artifact_policy_run_root("s3://bucket", run_name="run_a", epoch=None)
    assert isinstance(root, ArtifactReference)
    assert str(root.value) == "s3://bucket/run_a"


def test_artifact_policy_run_root_path(tmp_path: Path):
    base = tmp_path / "replays"
    root = artifact_policy_run_root(base, run_name="run_a", epoch=None)
    assert isinstance(root, ArtifactReference)
    assert root.as_path() == base / "run_a"


def test_artifact_simulation_root_adds_suite_and_name():
    base = ensure_artifact_reference("s3://bucket/replays")
    sim_root = artifact_simulation_root(base, suite="nav", name="maze")
    assert isinstance(sim_root, ArtifactReference)
    assert str(sim_root.value) == "s3://bucket/replays/nav/maze"


def test_artifact_reference_http_url(tmp_path: Path):
    local_root = ensure_artifact_reference(tmp_path)
    assert local_root is not None
    assert local_root.to_public_url() is None

    remote_root = ensure_artifact_reference("s3://bucket/path")
    assert remote_root is not None
    assert remote_root.to_public_url() == "https://bucket.s3.amazonaws.com/path"


def test_parsed_uri_bucket_root_has_no_key():
    parsed = ParsedURI.parse("s3://bucket")
    assert parsed.bucket == "bucket"
    assert parsed.key is None


def test_artifact_reference_with_policy_and_simulation_helpers():
    base = ensure_artifact_reference("s3://bucket/replays")
    assert base is not None
    policy_root = base.with_policy("run_a", 4)
    assert str(policy_root.value) == "s3://bucket/replays/run_a/v4"
    sim_root = policy_root.with_simulation("suite", "sim", simulation_id="abc123")
    assert str(sim_root.value) == "s3://bucket/replays/run_a/v4/suite/sim/abc123"


def test_ensure_artifact_reference_rejects_empty_strings():
    with pytest.raises(ValueError):
        ensure_artifact_reference("")
    with pytest.raises(ValueError):
        ensure_artifact_reference("   ")


def test_ensure_artifact_reference_strips_whitespace():
    ref = ensure_artifact_reference("  s3://bucket/path  ")
    assert isinstance(ref, ArtifactReference)
    assert str(ref.value) == "s3://bucket/path"
