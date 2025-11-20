"""CLI helpers for displaying leaderboard data."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Optional

import httpx
import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cli.login import DEFAULT_COGAMES_SERVER, CoGamesAuthenticator
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER


def _format_timestamp(value: Optional[str]) -> str:
    """Format ISO timestamps for CLI output."""
    if not value:
        return "—"
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        return value

    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone()
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _format_score(value: Any) -> str:
    """Format numeric scores for display."""
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "—"


def _load_token(login_server: str) -> str | None:
    authenticator = CoGamesAuthenticator()
    if not authenticator.has_saved_token(login_server):
        console.print("[red]Error:[/red] Not authenticated.")
        console.print("Please run: [cyan]cogames login[/cyan]")
        return None

    token = authenticator.load_token(login_server)
    if not token:
        console.print(f"[red]Error:[/red] Token not found for {login_server}")
        return None
    return token


def _fetch_leaderboard(server: str, path: str, token: str) -> dict[str, Any]:
    console.print(f"[cyan]Fetching leaderboard data from {path}...[/cyan]")
    try:
        response = httpx.get(
            f"{server}{path}",
            headers={"X-Auth-Token": token},
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach {server}:[/red] {exc}")
        raise typer.Exit(1) from exc

    if response.status_code != 200:
        console.print(f"[red]Request failed with status {response.status_code}[/red]")
        console.print(f"[dim]{response.text}[/dim]")
        raise typer.Exit(1)

    return response.json()


def _render_table(title: str, entries: list[dict[str, Any]]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Policy", style="bold cyan")
    table.add_column("Created", style="green")
    table.add_column("Avg Score", justify="right")
    table.add_column("Version ID", style="dim")

    for entry in entries:
        policy_version = entry.get("policy_version", {})
        policy_label = f"{policy_version.get('name', '?')}.{policy_version.get('version', '?')}"
        created_at = policy_version.get("policy_created_at") or policy_version.get("created_at")
        table.add_row(
            policy_label,
            _format_timestamp(created_at),
            _format_score(entry.get("avg_score")),
            str(policy_version.get("id", "—")),
        )

        scores = entry.get("scores") or {}
        for sim_name, score_value in sorted(scores.items()):
            pretty_name = sim_name.split(":", 1)[-1] if ":" in sim_name else sim_name
            table.add_row(f"    {pretty_name}", "", _format_score(score_value), "")

    console.print(table)


def submissions_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        help="Login/authentication server URL",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        "-s",
        help="Observatory API base URL",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the raw JSON response instead of a table",
    ),
) -> None:
    """Fetch submissions tied to the authenticated user."""
    token = _load_token(login_server)
    if not token:
        return

    payload = _fetch_leaderboard(server, "/leaderboard/v2/users/me", token)
    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    entries = payload.get("entries") or []
    if not entries:
        console.print("[yellow]No submissions found.[/yellow]")
        return

    _render_table("Your Submissions", entries)


def leaderboard_cmd(
    view: Literal["public", "mine"] = typer.Option(
        "public",
        "--view",
        "-v",
        help="Choose 'public' to view the global leaderboard or 'mine' for your submissions.",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        help="Login/authentication server URL",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        "-s",
        help="Observatory API base URL",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the raw JSON response instead of a table",
    ),
) -> None:
    """Display leaderboard entries (public or personal)."""
    token = _load_token(login_server)
    if not token:
        return

    path = "/leaderboard/v2" if view == "public" else "/leaderboard/v2/users/me"
    payload = _fetch_leaderboard(server, path, token)
    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    entries = payload.get("entries") or []
    if not entries:
        console.print("[yellow]No entries found for this view.[/yellow]")
        return

    table_title = "Public Leaderboard" if view == "public" else "My Leaderboard Entries"
    _render_table(table_title, entries)
