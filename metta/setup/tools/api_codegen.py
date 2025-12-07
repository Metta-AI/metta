import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import typer
from rich.console import Console

from metta.common.util.fs import get_repo_root

app = typer.Typer(help="API client code generation tools")
console = Console()


@app.command()
def generate():
    """Generate TypeScript API types from OpenAPI spec."""
    root = get_repo_root()
    openapi_path = root / "app_backend" / "openapi.json"
    api_types_path = root / "observatory" / "src" / "api-types.ts"

    console.print("[bold]==> Exporting OpenAPI spec...[/bold]")
    _export_openapi(openapi_path)

    console.print("[bold]==> Generating TypeScript types...[/bold]")
    _generate_typescript(root)

    console.print("[bold]==> Formatting with Prettier...[/bold]")
    _format_with_prettier(root, api_types_path)

    console.print("[bold green]==> Done! TypeScript types are up to date.[/bold green]")


@app.command()
def export_openapi():
    """Export OpenAPI spec from FastAPI app."""
    root = get_repo_root()
    openapi_path = root / "app_backend" / "openapi.json"
    _export_openapi(openapi_path)
    console.print(f"[green]Exported OpenAPI spec to {openapi_path}[/green]")


def _export_openapi(output_path: Path):
    from metta.app_backend.server import create_app

    mock_repo = MagicMock()
    fastapi_app = create_app(mock_repo)
    openapi_spec = fastapi_app.openapi()

    with open(output_path, "w") as f:
        json.dump(openapi_spec, f, indent=2)


def _generate_typescript(root: Path):
    observatory_dir = root / "observatory"
    result = subprocess.run(
        ["pnpm", "run", "generate-api-types"],
        cwd=observatory_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]TypeScript generation failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)
    console.print(result.stdout.strip())


def _format_with_prettier(root: Path, api_types_path: Path):
    """Run prettier on the generated file to ensure consistent formatting."""
    observatory_dir = root / "observatory"
    result = subprocess.run(
        ["pnpm", "exec", "prettier", "--write", str(api_types_path)],
        cwd=observatory_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Prettier formatting failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
