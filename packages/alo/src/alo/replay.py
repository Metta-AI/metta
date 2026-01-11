from __future__ import annotations

from typing import Protocol


class ReplayLike(Protocol):
    def set_compression(self, compression: str) -> None: ...

    def write_replay(self, path: str) -> None: ...


def write_replay(replay: ReplayLike, path: str) -> None:
    if path.endswith(".gz"):
        replay.set_compression("gzip")
    elif path.endswith(".z"):
        replay.set_compression("zlib")
    replay.write_replay(path)
