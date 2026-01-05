"""CLI helpers for displaying leaderboard data."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

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
        data = client._get("/stats/policies/my-versions")
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc

    entries = data.get("entries", [])

    if policy_name:
        name, version = parse_policy_identifier(policy_name)
        if version:
            entries = [e for e in entries if e.get("name") == name and e.get("version") == version]
        else:
            entries = [e for e in entries if e.get("name") == name]

    if json_output:
        console.print(json.dumps(entries, indent=2))
        return

    if not entries:
        if policy_name:
            console.print(f"[yellow]No uploads found matching '{policy_name}'.[/yellow]")
        else:
            console.print("[yellow]No uploads found.[/yellow]")
        return

    table = Table(title="Your Uploaded Policies", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Policy", style="bold cyan")
    table.add_column("Version", justify="right")
    table.add_column("Uploaded", style="green")
    table.add_column("ID", style="dim")

    for entry in entries:
        table.add_row(
            entry.get("name", "?"),
            str(entry.get("version", "?")),
            _format_timestamp(entry.get("created_at")),
            str(entry.get("id", "—")),
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
        entries = client._get(f"/tournament/seasons/{season}/policies")
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
    table.add_column("Pools", style="white")
    table.add_column("Matches", justify="right")
    table.add_column("Entered", style="green")

    for entry in entries:
        policy = entry.get("policy", {})
        pools = entry.get("pools", [])
        pool_names = [p.get("pool_name", "?") for p in pools]
        total_matches = sum(p.get("completed", 0) + p.get("failed", 0) + p.get("pending", 0) for p in pools)
        table.add_row(
            policy.get("name", "?"),
            str(policy.get("version", "?")),
            ", ".join(pool_names) if pool_names else "—",
            str(total_matches),
            _format_timestamp(entry.get("entered_at")),
        )

    console.print(table)


def leaderboard_cmd(
    season: str = typer.Option(
        ...,
        "--season",
        help="Tournament season name (required)",
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
    """Display the tournament leaderboard for a season.

    Example:
      cogames leaderboard --season beta
    """
    client = _get_client(login_server, server)
    if not client:
        return

    try:
        with client:
            entries = client._get(f"/tournament/seasons/{season}/leaderboard")
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

    if json_output:
        console.print(json.dumps(entries, indent=2))
        return

    if not entries:
        console.print(f"[yellow]No leaderboard entries for season '{season}'.[/yellow]")
        return

    table = Table(title=f"Leaderboard: {season}", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Policy", style="bold cyan")
    table.add_column("Version", justify="right")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Matches", justify="right")

    for entry in entries:
        policy = entry.get("policy", {})
        table.add_row(
            str(entry.get("rank", "?")),
            policy.get("name", "?"),
            str(policy.get("version", "?")),
            _format_score(entry.get("score")),
            str(entry.get("matches", 0)),
        )

    console.print(table)


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
