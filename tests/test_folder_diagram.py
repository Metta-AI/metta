from pathlib import Path

from metta.setup.tools.codebase import _generate_mermaid, _parse_folders

REPO_ROOT = Path(__file__).parent.parent
README_PATH = REPO_ROOT / "README.md"
MARKER_START = "<!-- FOLDER_DIAGRAM_START -->"
MARKER_END = "<!-- FOLDER_DIAGRAM_END -->"


def test_folder_diagram_is_up_to_date():
    """Verify the README folder diagram matches .importlinter.

    If this test fails, run: metta codebase generate-folder-diagram
    """
    folders = _parse_folders()
    expected_mermaid = _generate_mermaid(folders)

    readme_content = README_PATH.read_text()
    start_idx = readme_content.find(MARKER_START)
    end_idx = readme_content.find(MARKER_END)

    assert start_idx != -1, f"Marker '{MARKER_START}' not found in README.md"
    assert end_idx != -1, f"Marker '{MARKER_END}' not found in README.md"

    actual_content = readme_content[start_idx + len(MARKER_START) : end_idx].strip()

    assert actual_content == expected_mermaid.strip(), (
        "README folder diagram is out of date with .importlinter.\nRun: metta codebase generate-folder-diagram"
    )
