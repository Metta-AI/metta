import configparser
import re

import typer

from metta.common.util.fs import get_repo_root

app = typer.Typer(help="Codebase management tools.")

MARKER_START = "<!-- FOLDER_DIAGRAM_START -->"
MARKER_END = "<!-- FOLDER_DIAGRAM_END -->"


def _parse_folders() -> list[str]:
    repo_root = get_repo_root()
    importlinter_path = repo_root / ".importlinter"

    config = configparser.ConfigParser()
    config.read(importlinter_path)

    folders_section = "importlinter:contract:metta-folders"
    if folders_section not in config:
        raise typer.BadParameter(f"Section '{folders_section}' not found in .importlinter")

    layers_raw = config[folders_section].get("layers", "")
    return [line.strip() for line in layers_raw.strip().split("\n") if line.strip()]


def _generate_mermaid(folders: list[str]) -> str:
    lines = ["```mermaid", "graph TD"]

    short_names = {}
    for folder in folders:
        short = folder.replace("metta.", "")
        short_names[folder] = short

    for folder in folders:
        short = short_names[folder]
        lines.append(f"    {short}[{short}]")

    for i in range(len(folders) - 1):
        upper = short_names[folders[i]]
        lower = short_names[folders[i + 1]]
        lines.append(f"    {upper} --> {lower}")

    lines.append("```\n")
    return "\n".join(lines)


@app.command("generate-folder-diagram")
def generate_folder_diagram(
    print_only: bool = typer.Option(False, "--print", help="Print diagram without updating README"),
):
    """Generate folder hierarchy diagram from .importlinter and update README.md."""
    repo_root = get_repo_root()
    readme_path = repo_root / "README.md"

    folders = _parse_folders()
    mermaid = _generate_mermaid(folders)

    if print_only:
        typer.echo(mermaid)
        raise typer.Exit(0)

    content = readme_path.read_text()

    if MARKER_START not in content:
        typer.echo(f"Marker '{MARKER_START}' not found in README.md")
        typer.echo("Add the following to README.md where you want the diagram:")
        typer.echo(f"\n{MARKER_START}\n{MARKER_END}")
        raise typer.Exit(1)

    pattern = re.compile(
        rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
        re.DOTALL,
    )

    replacement = f"{MARKER_START}\n\n{mermaid}\n{MARKER_END}"
    new_content = pattern.sub(replacement, content)

    if new_content != content:
        readme_path.write_text(new_content)
        typer.echo("Updated README.md with folder diagram")
    else:
        typer.echo("README.md already up to date")
