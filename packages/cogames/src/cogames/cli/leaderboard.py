"""CLI helpers for displaying leaderboard data."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

import httpx
import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cli.client import TournamentServerClient
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER


def parse_policy_identifier(identifier: str) -> tuple[str, int | None]:
    """Parse 'name' or 'name:v3' into (name, version).

    Accepts formats:
    - 'my-policy' -> ('my-policy', None) - latest version
    - 'my-policy:v3' -> ('my-policy', 3) - specific version
    - 'my-policy:3' -> ('my-policy', 3) - specific version
    """
    if ":" in identifier:
        name, version_str = identifier.rsplit(":", 1)
        version_str = version_str.lstrip("v")
        try:
            version = int(version_str)
        except ValueError:
            raise ValueError(f"Invalid version format: {identifier}") from None
        return name, version
    return identifier, None


def lookup_policy_version_id(
    server: str,
    token: str,
    name: str,
    version: int | None,
) -> uuid.UUID | None:
    """Look up a policy version ID by name and optional version."""
    try:
        params = {"name": name}
        if version is not None:
            params["version"] = str(version)

        response = httpx.get(
            f"{server}/stats/policies/my-versions/lookup",
            params=params,
            headers={"X-Auth-Token": token},
            timeout=30.0,
        )
        if response.status_code == 200:
            data = response.json()
            return uuid.UUID(data["id"])
        elif response.status_code == 404:
            return None
        else:
            console.print(f"[red]Lookup failed with status {response.status_code}[/red]")
            console.print(f"[dim]{response.text}[/dim]")
            return None
    except Exception as exc:
        console.print(f"[red]Lookup failed:[/red] {exc}")
        return None


def submit_to_season(
    server: str,
    token: str,
    season: str,
    policy_version_id: uuid.UUID,
) -> list[str] | None:
    """Submit a policy version to a tournament season. Returns list of pool names on success."""
    try:
        response = httpx.post(
            f"{server}/tournament/seasons/{season}/submissions",
            json={"policy_version_id": str(policy_version_id)},
            headers={"X-Auth-Token": token},
            timeout=60.0,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("pools", [])
        elif response.status_code == 404:
            console.print(f"[red]Season '{season}' not found[/red]")
            return None
        else:
            console.print(f"[red]Submit failed with status {response.status_code}[/red]")
            console.print(f"[dim]{response.text}[/dim]")
            return None
    except Exception as exc:
        console.print(f"[red]Submit failed:[/red] {exc}")
        return None


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


def _get_client(login_server: str, server: str) -> TournamentServerClient | None:
    return TournamentServerClient.from_login(server_url=server, login_server=login_server)


def _fetch_leaderboard(client: TournamentServerClient, path: str) -> dict[str, Any]:
    try:
        return client._get(path)
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc


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
    client = _get_client(login_server, server)
    if not client:
        return

    with client:
        payload = _fetch_leaderboard(client, "/leaderboard/v2/users/me")

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
    client = _get_client(login_server, server)
    if not client:
        return

    path = "/leaderboard/v2" if view == "public" else "/leaderboard/v2/users/me"
    with client:
        payload = _fetch_leaderboard(client, path)

    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    entries = payload.get("entries") or []
    if not entries:
        console.print("[yellow]No entries found for this view.[/yellow]")
        return

    table_title = "Public Leaderboard" if view == "public" else "My Leaderboard Entries"
    _render_table(table_title, entries)


def seasons_cmd(
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
    """List available tournament seasons."""
    client = _get_client(login_server, server)
    if not client:
        return

    try:
        with client:
            seasons = client.get_seasons()
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach {server}:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        console.print(json.dumps([s.model_dump() for s in seasons], indent=2))
        return

    if not seasons:
        console.print("[yellow]No seasons found.[/yellow]")
        return

    table = Table(title="Tournament Seasons", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Season", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_column("Pools", style="dim")

    for season in seasons:
        pool_names = [p.name for p in season.pools]
        pools_str = ", ".join(pool_names) if pool_names else "—"
        table.add_row(season.name, season.summary, pools_str)

    console.print(table)
