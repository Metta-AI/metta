import tempfile
from pathlib import Path

from codeclip.file import get_context_objects


def test_get_context_objects_returns_files():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.py"
        p.write_text("print('hi')\n")
        ctx = get_context_objects(paths=[td])
        assert p.as_posix() in ctx.files
        assert "print('hi')" in ctx.files[p.as_posix()]
        assert ctx.total_files >= 1
        assert ctx.total_tokens > 0


def test_get_context_objects_empty_paths():
    ctx = get_context_objects(paths=None)
    assert ctx.files == {}
    assert ctx.total_files == 0
    assert ctx.total_tokens == 0
    assert len(ctx.documents) == 0


def test_get_context_objects_returns_context_object():
    """Test that get_context_objects returns a CodeContext with the expected structure."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.py"
        content = "def hello():\n    return 'world'\n"
        p.write_text(content)

        ctx = get_context_objects(paths=[td])

        # Check CodeContext structure
        assert hasattr(ctx, "documents")
        assert hasattr(ctx, "files")
        assert hasattr(ctx, "total_tokens")
        assert hasattr(ctx, "total_files")
        assert hasattr(ctx, "path_summaries")
        assert hasattr(ctx, "file_token_counts")
        assert hasattr(ctx, "top_level_summary")

        # Check that files and documents are consistent
        assert len(ctx.documents) == len(ctx.files)
        assert ctx.total_files == len(ctx.files)

        # Check that the content matches
        file_path = p.as_posix()
        assert file_path in ctx.files
        assert ctx.files[file_path] == content
