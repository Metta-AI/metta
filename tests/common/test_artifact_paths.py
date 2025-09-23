from pathlib import Path

import pytest

from mettagrid.util.uri import (
    ParsedURI,
    artifact_join,
    artifact_policy_run_root,
    artifact_simulation_root,
)


def test_artifact_join_s3():
    joined = artifact_join("s3://bucket/replays", "run_a", "sim", "episode.json.z")
    assert joined == "s3://bucket/replays/run_a/sim/episode.json.z"


def test_artifact_join_s3_with_trailing_slash():
    joined = artifact_join("s3://bucket/replays/", "run_a")
    assert joined == "s3://bucket/replays/run_a"


def test_artifact_join_s3_bucket_root():
    joined = artifact_join("s3://bucket", "run_a", "episode.json.z")
    assert joined == "s3://bucket/run_a/episode.json.z"


def test_artifact_join_s3_bucket_root_trailing_slash():
    joined = artifact_join("s3://bucket/", "run_a")
    assert joined == "s3://bucket/run_a"


def test_artifact_join_local_str(tmp_path: Path):
    joined = artifact_join(tmp_path / "replays", "run_a", "sim")
    assert Path(joined) == tmp_path / "replays" / "run_a" / "sim"


def test_artifact_join_path(tmp_path: Path):
    joined = artifact_join(tmp_path / "policies", "run_a", "checkpoints")
    assert Path(joined) == tmp_path / "policies" / "run_a" / "checkpoints"


def test_artifact_join_gdrive():
    joined = artifact_join("gdrive://folder", "run_a", "episode.json.z")
    assert joined == "gdrive://folder/run_a/episode.json.z"


def test_artifact_policy_run_root_s3_epoch():
    root = artifact_policy_run_root("s3://bucket/replays", run_name="run_a", epoch=3)
    assert root == "s3://bucket/replays/run_a/v3"


def test_artifact_policy_run_root_s3_bucket_root():
    root = artifact_policy_run_root("s3://bucket", run_name="run_a", epoch=None)
    assert root == "s3://bucket/run_a"


def test_artifact_policy_run_root_path(tmp_path: Path):
    root = artifact_policy_run_root(tmp_path / "replays", run_name="run_a", epoch=None)
    assert Path(root) == tmp_path / "replays" / "run_a"


def test_artifact_simulation_root_adds_suite_and_name():
    sim_root = artifact_simulation_root("s3://bucket/replays/run", suite="nav", name="maze")
    assert sim_root == "s3://bucket/replays/run/nav/maze"


def test_artifact_join_rejects_empty_s3_bucket():
    with pytest.raises(ValueError):
        artifact_join("s3://", "foo")


def test_parsed_uri_bucket_root_has_no_key():
    parsed = ParsedURI.parse("s3://bucket")
    assert parsed.bucket == "bucket"
    assert parsed.key is None
