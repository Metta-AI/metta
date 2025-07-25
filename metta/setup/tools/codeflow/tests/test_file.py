import os
import tempfile
import unittest
from pathlib import Path

from codeflow.file import (
    _find_git_root,
    _find_gitignore,
    _find_parent_readmes,
    _read_gitignore,
    _should_ignore,
    _should_ignore_file_type,
    resolve_codebase_path,
)


class TestGitignoreHandling(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

        structure = {
            "test.py": "",
            "test.pyc": "",
            "lib.so": "",
            "subdir": {
                "nested.py": "",
                "nested.pyc": "",
                "special.txt": "",
                "special.py": "",
                ".gitignore": "!special.py\n*.pyc\n",
            },
            "build": {
                "build.py": "",
                "lib.so": "",
            },
            "dist": {
                "dist.so": "",
            },
            "wandb": {
                "data.txt": "",
            },
            "train_dir": {
                "model.pt": "",
            },
            "outputs": {
                "output.log": "",
            },
            "cython_debug": {
                "debug.log": "",
            },
            ".codeflowignore": (
                "*.pyc\n__pycache__/\n/wandb/\ntrain_dir/\nreplays/\n*.egg-info\n"
                "/build/\n/build_debug/\n.DS_Store\n.task\noutputs/\n*.so\n"
                "cython_debug\nstats.profile\n/dist/\nplayer/dist/\nplayer/node_modules\n"
            ),
        }
        self._create_structure(self.base_path, structure)

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_structure(self, base, structure):
        for name, content in structure.items():
            path = base / name
            if isinstance(content, dict):
                path.mkdir(parents=True, exist_ok=True)
                self._create_structure(path, content)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)

    def test_read_gitignore(self):
        """Test reading gitignore patterns from file."""
        gitignore_path = self.base_path / ".codeflowignore"
        rules = _read_gitignore(str(gitignore_path))

        # Check that patterns were read correctly
        self.assertIn("*.pyc", rules)
        self.assertIn("__pycache__/", rules)
        self.assertIn("/wandb/", rules)
        self.assertIn("train_dir/", rules)
        self.assertIn("*.so", rules)

    def test_should_ignore_patterns(self):
        """Test the ignore logic for various file patterns."""
        gitignore_rules = _read_gitignore(str(self.base_path / ".codeflowignore"))

        # Binary files should be ignored due to extension
        self.assertTrue(_should_ignore(self.base_path / "test.pyc", gitignore_rules, self.base_path))
        self.assertTrue(_should_ignore(self.base_path / "lib.so", gitignore_rules, self.base_path))
        self.assertTrue(_should_ignore(self.base_path / "subdir/nested.pyc", gitignore_rules, self.base_path))
        self.assertTrue(_should_ignore(self.base_path / "build/lib.so", gitignore_rules, self.base_path))
        self.assertTrue(_should_ignore(self.base_path / "dist/dist.so", gitignore_rules, self.base_path))

        # Data files should be ignored due to extension
        self.assertTrue(_should_ignore(self.base_path / "outputs/output.log", gitignore_rules, self.base_path))
        self.assertTrue(_should_ignore(self.base_path / "cython_debug/debug.log", gitignore_rules, self.base_path))

        # Paths that should not be ignored
        self.assertFalse(_should_ignore(self.base_path / "test.py", gitignore_rules, self.base_path))
        self.assertFalse(_should_ignore(self.base_path / "subdir/nested.py", gitignore_rules, self.base_path))

    def test_readme_never_ignored(self):
        """Test that README.md files are never ignored."""
        gitignore_rules = ["*.md", "README.md"]

        # README.md should never be ignored even if explicitly in gitignore
        self.assertFalse(_should_ignore(self.base_path / "README.md", gitignore_rules, self.base_path))
        self.assertFalse(_should_ignore(self.base_path / "subdir/README.md", gitignore_rules, self.base_path))

    def test_gitignore_with_lock_file(self):
        """Test that lock files in gitignore are properly ignored."""
        # Create a test directory with .gitignore containing uv.lock
        test_dir = self.base_path / "lock_test"
        test_dir.mkdir()

        # Create .gitignore with uv.lock
        gitignore_path = test_dir / ".gitignore"
        with open(gitignore_path, "w") as f:
            f.write("uv.lock\n*.pyc\n__pycache__/\n")

        # Create uv.lock file
        lock_file = test_dir / "uv.lock"
        with open(lock_file, "w") as f:
            f.write("# This is a lock file with many tokens\n" * 1000)

        # Read gitignore rules
        gitignore_rules = _read_gitignore(str(gitignore_path))

        # Verify uv.lock is in the rules
        self.assertIn("uv.lock", gitignore_rules)

        # Test that uv.lock should be ignored
        self.assertTrue(_should_ignore(lock_file, gitignore_rules, test_dir))

    def test_read_gitignore_with_directory_path(self):
        """Test that _read_gitignore fails when given a directory path instead of file path."""
        # This test demonstrates the bug: _read_gitignore expects a file path
        # but get_context passes it a directory path

        # Create a test directory with .gitignore
        test_dir = self.base_path / "gitignore_test"
        test_dir.mkdir()

        # Create .gitignore file
        gitignore_path = test_dir / ".gitignore"
        with open(gitignore_path, "w") as f:
            f.write("uv.lock\n*.pyc\n")

        # Test 1: Correct usage - pass the .gitignore file path
        rules_correct = _read_gitignore(str(gitignore_path))
        self.assertIn("uv.lock", rules_correct)
        self.assertIn("*.pyc", rules_correct)

        # Test 2: Incorrect usage (what get_context does) - pass directory path
        rules_incorrect = _read_gitignore(str(test_dir))
        self.assertEqual(
            rules_incorrect, [], "Passing directory path to _read_gitignore returns empty list - this is the bug!"
        )


class TestGitRootAndReadmeHandling(unittest.TestCase):
    """Test cases for git root detection and parent README finding."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_find_git_root(self):
        """Test finding git repository root."""
        # Create a directory structure with nested git repo
        repo_root = self.base_path / "myproject"
        repo_root.mkdir()
        (repo_root / ".git").mkdir()

        subdir = repo_root / "src" / "module"
        subdir.mkdir(parents=True)

        # Test from various locations
        self.assertEqual(_find_git_root(subdir), repo_root)
        self.assertEqual(_find_git_root(repo_root / "src"), repo_root)
        self.assertEqual(_find_git_root(repo_root), repo_root)

        # Test when not in a git repo
        non_git_dir = self.base_path / "non_git"
        non_git_dir.mkdir()
        self.assertIsNone(_find_git_root(non_git_dir))

    def test_find_parent_readmes_in_git_repo(self):
        """Test finding parent READMEs up to git root."""
        # Create a git repo structure
        repo_root = self.base_path / "myproject"
        repo_root.mkdir()
        (repo_root / ".git").mkdir()

        # Create READMEs at various levels
        (repo_root / "README.md").write_text("Root readme")

        src_dir = repo_root / "src"
        src_dir.mkdir()
        (src_dir / "README.md").write_text("Src readme")

        module_dir = src_dir / "module"
        module_dir.mkdir()
        (module_dir / "README.md").write_text("Module readme")

        # Create a README outside the git repo (should not be included)
        (self.base_path / "README.md").write_text("Outside readme")

        # Test from deep directory
        readmes = _find_parent_readmes(module_dir / "subdir")
        readme_paths = [str(r.relative_to(repo_root)) for r in readmes]

        # Should include READMEs up to git root, but not beyond
        self.assertIn("README.md", readme_paths)
        self.assertIn("src/README.md", readme_paths)
        self.assertIn("src/module/README.md", readme_paths)
        self.assertEqual(len(readmes), 3)

    def test_find_parent_readmes_without_git(self):
        """Test finding parent READMEs when not in a git repo."""
        # Create directory structure without git
        project_dir = self.base_path / "project"
        project_dir.mkdir()

        subdir = project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        # Create some READMEs
        (project_dir / "README.md").write_text("Project readme")
        (project_dir / "src" / "README.md").write_text("Src readme")

        # Test - should go all the way to root
        readmes = _find_parent_readmes(subdir)

        # Should find the READMEs we created
        self.assertEqual(len(readmes), 2)
        self.assertTrue(any("project/README.md" in str(r) for r in readmes))
        self.assertTrue(any("project/src/README.md" in str(r) for r in readmes))


class TestPathResolution(unittest.TestCase):
    """Test cases for path resolution relative to pwd."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.test_dir.cleanup()

    def test_relative_path_resolution(self):
        """Test that relative paths are resolved against pwd."""
        # Create a subdirectory
        subdir = Path("subdir")
        subdir.mkdir()

        # Test resolving relative path
        resolved = resolve_codebase_path("subdir")
        expected = (Path(self.test_dir.name) / "subdir").resolve()
        self.assertEqual(resolved, expected)

    def test_absolute_path_resolution(self):
        """Test that absolute paths are returned as-is."""
        abs_path = Path("/tmp/test/path").resolve()
        resolved = resolve_codebase_path(abs_path)
        self.assertEqual(resolved, abs_path)

    def test_dot_path_resolution(self):
        """Test that '.' resolves to current directory."""
        resolved = resolve_codebase_path(".")
        expected = Path(self.test_dir.name).resolve()
        self.assertEqual(resolved, expected)

    def test_parent_path_resolution(self):
        """Test that '..' resolves correctly."""
        # Create and change to subdirectory
        subdir = Path("subdir")
        subdir.mkdir()
        os.chdir(subdir)

        resolved = resolve_codebase_path("..")
        expected = Path(self.test_dir.name).resolve()
        self.assertEqual(resolved, expected)

    def test_nested_relative_path(self):
        """Test nested relative path resolution."""
        # Create nested structure
        nested = Path("a/b/c")
        nested.mkdir(parents=True)

        resolved = resolve_codebase_path("a/b/c")
        expected = (Path(self.test_dir.name) / "a" / "b" / "c").resolve()
        self.assertEqual(resolved, expected)

    def test_path_with_symlinks(self):
        """Test that paths with symlinks are properly resolved."""
        # Create a directory and a symlink to it
        real_dir = Path("real_dir")
        real_dir.mkdir()
        link_dir = Path("link_dir")
        link_dir.symlink_to(real_dir)

        # Resolve through symlink
        resolved = resolve_codebase_path("link_dir")
        # Should resolve to the actual directory
        self.assertTrue(resolved.exists())
        self.assertTrue(resolved.is_dir())

    def test_nonexistent_path_resolution(self):
        """Test that nonexistent paths still resolve correctly."""
        # Path doesn't need to exist to be resolved
        resolved = resolve_codebase_path("nonexistent/path")
        expected = (Path(self.test_dir.name) / "nonexistent" / "path").resolve()
        self.assertEqual(resolved, expected)


class TestFileTypeIgnoring(unittest.TestCase):
    """Test cases for the unified file type ignoring system."""

    def test_binary_extensions(self):
        """Test that binary file extensions are properly ignored."""
        # Python compiled
        self.assertTrue(_should_ignore_file_type(Path("test.pyc")))
        self.assertTrue(_should_ignore_file_type(Path("module.pyo")))
        self.assertTrue(_should_ignore_file_type(Path("lib.pyd")))

        # Native libraries
        self.assertTrue(_should_ignore_file_type(Path("lib.so")))
        self.assertTrue(_should_ignore_file_type(Path("module.dll")))
        self.assertTrue(_should_ignore_file_type(Path("lib.dylib")))

        # ML model files
        self.assertTrue(_should_ignore_file_type(Path("model.pt")))
        self.assertTrue(_should_ignore_file_type(Path("checkpoint.ckpt")))
        self.assertTrue(_should_ignore_file_type(Path("weights.safetensors")))

        # Database files
        self.assertTrue(_should_ignore_file_type(Path("data.db")))
        self.assertTrue(_should_ignore_file_type(Path("app.sqlite")))
        self.assertTrue(_should_ignore_file_type(Path("test.sqlite3")))

        # Images
        self.assertTrue(_should_ignore_file_type(Path("image.jpg")))
        self.assertTrue(_should_ignore_file_type(Path("photo.png")))
        self.assertTrue(_should_ignore_file_type(Path("icon.svg")))

    def test_data_extensions(self):
        """Test that data file extensions are properly ignored."""
        self.assertTrue(_should_ignore_file_type(Path("app.log")))
        self.assertTrue(_should_ignore_file_type(Path("data.cache")))
        self.assertTrue(_should_ignore_file_type(Path("temp.tmp")))
        self.assertTrue(_should_ignore_file_type(Path("backup.bak")))
        self.assertTrue(_should_ignore_file_type(Path(".file.swp")))

    def test_lock_files(self):
        """Test that lock files are properly ignored."""
        self.assertTrue(_should_ignore_file_type(Path("uv.lock")))
        self.assertTrue(_should_ignore_file_type(Path("package-lock.json")))
        self.assertTrue(_should_ignore_file_type(Path("yarn.lock")))
        self.assertTrue(_should_ignore_file_type(Path("poetry.lock")))
        self.assertTrue(_should_ignore_file_type(Path("Cargo.lock")))

    def test_cache_and_system_files(self):
        """Test that cache and system files are ignored."""
        self.assertTrue(_should_ignore_file_type(Path(".DS_Store")))
        self.assertTrue(_should_ignore_file_type(Path("Thumbs.db")))
        self.assertTrue(_should_ignore_file_type(Path("desktop.ini")))
        self.assertTrue(_should_ignore_file_type(Path(".coverage")))
        self.assertTrue(_should_ignore_file_type(Path("coverage.xml")))

    def test_file_patterns(self):
        """Test that file patterns are properly matched."""
        self.assertTrue(_should_ignore_file_type(Path("gpu_names_cache.json")))
        self.assertTrue(_should_ignore_file_type(Path("package.egg-info")))
        self.assertTrue(_should_ignore_file_type(Path("file~")))  # Editor backup
        self.assertTrue(_should_ignore_file_type(Path(".#lockfile")))  # Emacs lock

    def test_allowed_files(self):
        """Test that source files are not ignored."""
        self.assertFalse(_should_ignore_file_type(Path("main.py")))
        self.assertFalse(_should_ignore_file_type(Path("README.md")))
        self.assertFalse(_should_ignore_file_type(Path("config.yaml")))
        self.assertFalse(_should_ignore_file_type(Path("script.js")))
        self.assertFalse(_should_ignore_file_type(Path("style.css")))
        self.assertFalse(_should_ignore_file_type(Path("doc.txt")))  # .txt is allowed now

    def test_case_insensitive_extensions(self):
        """Test that extension checking is case-insensitive."""
        self.assertTrue(_should_ignore_file_type(Path("IMAGE.JPG")))
        self.assertTrue(_should_ignore_file_type(Path("Model.PT")))
        self.assertTrue(_should_ignore_file_type(Path("data.LOG")))


class TestGitignoreFinding(unittest.TestCase):
    """Test cases for finding gitignore files in the directory tree."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_find_gitignore_in_current_dir(self):
        """Test finding .gitignore in the current directory."""
        project_dir = self.base_path / "project"
        project_dir.mkdir()
        gitignore_path = project_dir / ".gitignore"
        gitignore_path.write_text("*.pyc\n")

        found = _find_gitignore(project_dir)
        self.assertEqual(found, gitignore_path)

    def test_find_gitignore_in_parent(self):
        """Test finding .gitignore in a parent directory."""
        project_dir = self.base_path / "project"
        project_dir.mkdir()
        gitignore_path = project_dir / ".gitignore"
        gitignore_path.write_text("*.pyc\n")

        subdir = project_dir / "src" / "module"
        subdir.mkdir(parents=True)

        found = _find_gitignore(subdir)
        self.assertEqual(found, gitignore_path)

    def test_find_gitignore_with_git_root(self):
        """Test that search stops at git root."""
        # Create a git repo
        repo_dir = self.base_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        # Create .gitignore outside repo (should not be found)
        outer_gitignore = self.base_path / ".gitignore"
        outer_gitignore.write_text("outer\n")

        # Create subdir in repo
        subdir = repo_dir / "src"
        subdir.mkdir()

        # Should not find the outer gitignore
        found = _find_gitignore(subdir)
        self.assertIsNone(found)

        # Add gitignore inside repo
        repo_gitignore = repo_dir / ".gitignore"
        repo_gitignore.write_text("inner\n")

        # Now should find the repo's gitignore
        found = _find_gitignore(subdir)
        self.assertEqual(found, repo_gitignore)


if __name__ == "__main__":
    unittest.main()
