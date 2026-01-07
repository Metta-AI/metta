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
    policy_name: Optional[str] = typer.Argument(
        None,
        help="Policy name to filter (e.g., 'my-policy' or 'my-policy:v3')",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        help="Filter by tournament season",
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
    """Show your uploaded policies and tournament submissions.

    Examples:
      cogames submissions                    # All uploads
      cogames submissions --season beta      # Submissions in beta season
      cogames submissions my-policy          # Info on a specific policy
    """
    client = _get_client(login_server, server)
    if not client:
        return

    with client:
        if season:
            _show_season_submissions(client, season, policy_name, json_output)
        else:
            _show_all_uploads(client, policy_name, json_output)


def _show_all_uploads(
    client: TournamentServerClient,
    policy_name: Optional[str],
    json_output: bool,
) -> None:
    """Show all uploaded policies."""
    try:
        name_filter = None
        version_filter = None
        if policy_name:
            name_filter, version_filter = parse_policy_identifier(policy_name)
        entries = client.get_my_policy_versions(name=name_filter, version=version_filter)
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        console.print(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return

    if not entries:
        if policy_name:
            console.print(f"[yellow]No uploads found matching '{policy_name}'.[/yellow]")
        else:
            console.print("[yellow]No uploads found.[/yellow]")
        return

    try:
        memberships = client.get_my_memberships()
    except httpx.HTTPError:
        memberships = {}

    table = Table(title="Your Uploaded Policies", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Policy", style="bold cyan")
    table.add_column("Version", justify="right")
    table.add_column("Uploaded", style="green")
    table.add_column("Seasons", style="white")

    for entry in entries:
        seasons = memberships.get(str(entry.id), [])
        seasons_str = ", ".join(sorted(seasons)) if seasons else "—"
        table.add_row(
            entry.name,
            str(entry.version),
            _format_timestamp(entry.created_at.isoformat()),
            seasons_str,
        )

    console.print(table)


def _show_season_submissions(
    client: TournamentServerClient,
    season: str,
    policy_name: Optional[str],
    json_output: bool,
) -> None:
    """Show submissions for a specific season."""
    try:
        entries = client._get(f"/tournament/seasons/{season}/policies", params={"mine": "true"})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season}' not found[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[red]Request failed with status {exc.response.status_code}[/red]")
        console.print(f"[dim]{exc.response.text}[/dim]")
        raise typer.Exit(1) from exc
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc

    if policy_name:
        name, version = parse_policy_identifier(policy_name)
        if version:
            entries = [
                e
                for e in entries
                if e.get("policy", {}).get("name") == name and e.get("policy", {}).get("version") == version
            ]
        else:
            entries = [e for e in entries if e.get("policy", {}).get("name") == name]

    if json_output:
        console.print(json.dumps(entries, indent=2))
        return

    if not entries:
        if policy_name:
            console.print(f"[yellow]No submissions found for '{policy_name}' in season '{season}'.[/yellow]")
        else:
            console.print(f"[yellow]No submissions found in season '{season}'.[/yellow]")
        return

    table = Table(title=f"Submissions in '{season}'", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Policy", style="bold cyan")
    table.add_column("Version", justify="right")
    table.add_column("Pool", style="white")
    table.add_column("Status", style="dim")
    table.add_column("Matches", justify="right")
    table.add_column("Entered", style="green")

    for entry in entries:
        policy = entry.get("policy", {})
        pools = entry.get("pools", [])
        entered_at = _format_timestamp(entry.get("entered_at"))

        for i, p in enumerate(pools):
            pool_name = p.get("pool_name", "?")
            status = "active" if p.get("active", True) else "retired"
            matches = p.get("completed", 0) + p.get("failed", 0) + p.get("pending", 0)

            if i == 0:
                table.add_row(
                    policy.get("name", "?"),
                    str(policy.get("version", "?")),
                    pool_name,
                    status,
                    str(matches),
                    entered_at,
                )
            else:
                table.add_row("", "", pool_name, status, str(matches), "")

    console.print(table)


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
    try:
        with client:
            payload = client._get(path)
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc

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
        console.print(f"[red]Failed to reach server:[/red] {exc}")
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
