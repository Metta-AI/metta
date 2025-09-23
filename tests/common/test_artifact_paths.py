from pathlib import Path

from mettagrid.util.artifact_paths import artifact_path_join


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
