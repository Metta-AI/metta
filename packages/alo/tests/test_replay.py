from __future__ import annotations

from typing import Optional

from alo.replay import write_replay


class DummyReplay:
    def __init__(self) -> None:
        self.compression: Optional[str] = None
        self.path: Optional[str] = None

    def set_compression(self, compression: str) -> None:
        self.compression = compression

    def write_replay(self, path: str) -> None:
        self.path = path


def test_write_replay_sets_gzip() -> None:
    replay = DummyReplay()

    write_replay(replay, "replay.json.gz")

    assert replay.compression == "gzip"
    assert replay.path == "replay.json.gz"


def test_write_replay_sets_zlib() -> None:
    replay = DummyReplay()

    write_replay(replay, "replay.json.z")

    assert replay.compression == "zlib"
    assert replay.path == "replay.json.z"


def test_write_replay_no_compression() -> None:
    replay = DummyReplay()

    write_replay(replay, "replay.json")

    assert replay.compression is None
    assert replay.path == "replay.json"
