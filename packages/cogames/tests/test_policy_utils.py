"""Tests for policy utils remote checkpoint resolution."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from cogames.aws_storage import DownloadOutcome
from cogames.policy.utils import resolve_policy_data_path


def test_resolve_policy_download_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target_path = tmp_path / "remote.pt"

    def fake_download(*, policy_path: Path, **_: object) -> DownloadOutcome:
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text("stub")
        return DownloadOutcome(downloaded=True, reason="downloaded", details="s3://bucket/key")

    monkeypatch.setattr("cogames.policy.utils.maybe_download_checkpoint", fake_download)
    console = Console(file=io.StringIO())

    resolved = resolve_policy_data_path(
        str(target_path),
        policy_class_path="cogames.policy.simple.SimplePolicy",
        game_name="machina_1",
        console=console,
    )

    assert resolved == str(target_path)
    assert target_path.exists()


def test_resolve_policy_download_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target_path = tmp_path / "remote.pt"

    def fake_download(*, policy_path: Path, **_: object) -> DownloadOutcome:
        assert policy_path == target_path
        return DownloadOutcome(downloaded=False, reason="not_found", details=None)

    monkeypatch.setattr("cogames.policy.utils.maybe_download_checkpoint", fake_download)
    console = Console(file=io.StringIO())

    with pytest.raises(FileNotFoundError):
        resolve_policy_data_path(
            str(target_path),
            policy_class_path="cogames.policy.simple.SimplePolicy",
            game_name="machina_1",
            console=console,
        )

    assert not target_path.exists()
