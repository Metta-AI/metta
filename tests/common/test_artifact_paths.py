from pathlib import Path

from mettagrid.util.artifact_paths import artifact_path_join, artifact_policy_run_root


def test_artifact_path_join_s3():
    base = "s3://bucket/replays"
    joined = artifact_path_join(base, "run_a", "sim", "episode.json.z")
    assert joined == "s3://bucket/replays/run_a/sim/episode.json.z"


def test_artifact_path_join_s3_with_trailing_slash():
    base = "s3://bucket/replays/"
    joined = artifact_path_join(base, "run_a")
    assert joined == "s3://bucket/replays/run_a"


def test_artifact_path_join_local_str(tmp_path: Path):
    base = str(tmp_path / "replays")
    joined = artifact_path_join(base, "run_a", "sim")
    assert Path(joined) == tmp_path / "replays" / "run_a" / "sim"


def test_artifact_path_join_path(tmp_path: Path):
    base = tmp_path / "policies"
    joined = artifact_path_join(base, "run_a", "checkpoints")
    assert joined == tmp_path / "policies" / "run_a" / "checkpoints"


def test_artifact_path_join_gdrive():
    base = "gdrive://folder"
    joined = artifact_path_join(base, "run_a", "episode.json.z")
    assert joined == "gdrive://folder/run_a/episode.json.z"


def test_artifact_policy_run_root_s3_epoch():
    root = artifact_policy_run_root("s3://bucket/replays", run_name="run_a", epoch=3)
    assert root == "s3://bucket/replays/run_a/v3"


def test_artifact_policy_run_root_path(tmp_path: Path):
    base = tmp_path / "replays"
    root = artifact_policy_run_root(base, run_name="run_a", epoch=None)
    assert root == base / "run_a"
