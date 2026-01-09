from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from rich.console import Console

from cogames.cli.submit import _maybe_resolve_checkpoint_bundle_uri, validate_paths


def test_validate_paths_accepts_absolute_within_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights.safetensors").write_text("ok")
    console = Console()

    rel = validate_paths(["weights.safetensors"], console=console)
    assert rel == [Path("weights.safetensors")]

    abs_path = str((tmp_path / "weights.safetensors").resolve())
    rel2 = validate_paths([abs_path], console=console)
    assert rel2 == [Path("weights.safetensors")]


def test_validate_paths_rejects_outside_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    console = Console()

    with pytest.raises(ValueError):
        validate_paths([str(outside)], console=console)


def test_bundle_uri_directory_is_zipped(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "policy_spec.json").write_text(
        '{"class_path": "foo.Bar", "data_path": "weights.safetensors", "init_kwargs": {}}'
    )
    (bundle_dir / "weights.safetensors").write_text("ok")

    zip_path = None
    try:
        resolved = _maybe_resolve_checkpoint_bundle_uri(bundle_dir.as_uri())
        assert resolved is not None
        zip_path, cleanup = resolved
        assert cleanup is True
        assert zip_path.exists()

        with zipfile.ZipFile(zip_path) as zf:
            assert set(zf.namelist()) == {"policy_spec.json", "weights.safetensors"}
    finally:
        if zip_path and zip_path.exists():
            zip_path.unlink()


def test_bundle_uri_zip_is_reused(tmp_path: Path) -> None:
    bundle_zip = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.writestr("policy_spec.json", '{"class_path": "foo.Bar", "data_path": null, "init_kwargs": {}}')

    resolved = _maybe_resolve_checkpoint_bundle_uri(bundle_zip.as_uri())
    assert resolved == (bundle_zip, False)
