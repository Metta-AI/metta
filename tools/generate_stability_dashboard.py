#!/usr/bin/env python3
"""Generate a comprehensive stability dashboard for tracking stable tag history."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click


def run_git_command(cmd: List[str]) -> str:
    """Run a git command and return the output."""
    result = subprocess.run(["git"] + cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_stable_tags() -> List[Dict[str, Any]]:
    """Get all stable tags with their metadata."""
    tags = []
    
    # Get all tags matching stable-* pattern
    tag_list = run_git_command(["tag", "-l", "stable-*"]).split("\n")
    tag_list = [t for t in tag_list if t]  # Filter empty strings
    
    for tag in tag_list:
        # Get tag info
        tag_info = run_git_command(["show", "--no-patch", "--format=%H|%ai|%s", tag]).split("|")
        commit_hash = tag_info[0]
        timestamp = tag_info[1]
        subject = tag_info[2] if len(tag_info) > 2 else ""
        
        # Get tag message
        try:
            message = run_git_command(["tag", "-l", "--format=%(contents)", tag])
        except subprocess.CalledProcessError:
            message = ""
        
        tags.append({
            "tag": tag,
            "commit": commit_hash,
            "timestamp": timestamp,
            "subject": subject,
            "message": message
        })
    
    # Sort by timestamp (newest first)
    tags.sort(key=lambda x: x["timestamp"], reverse=True)
    return tags


def calculate_tag_intervals(tags: List[Dict[str, Any]]) -> List[float]:
    """Calculate time intervals between stable tags in days."""
    intervals = []
    
    for i in range(len(tags) - 1):
        current = datetime.fromisoformat(tags[i]["timestamp"].replace(" ", "T"))
        previous = datetime.fromisoformat(tags[i + 1]["timestamp"].replace(" ", "T"))
        interval_days = (current - previous).total_seconds() / 86400
        intervals.append(interval_days)
    
    return intervals


def get_workflow_runs(workflow_name: str = "advance-stable-tag.yml", limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent workflow runs for the stable tag advancement workflow."""
    try:
        # Use GitHub CLI if available
        cmd = [
            "gh", "run", "list",
            "--workflow", workflow_name,
            "--limit", str(limit),
            "--json", "conclusion,createdAt,headBranch,headSha,status,databaseId"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # GitHub CLI not available or command failed
        return []


def calculate_success_rate(runs: List[Dict[str, Any]]) -> float:
    """Calculate the success rate of workflow runs."""
    if not runs:
        return 0.0
    
    successful = sum(1 for run in runs if run.get("conclusion") == "success")
    return (successful / len(runs)) * 100


def generate_dashboard_html(
    tags: List[Dict[str, Any]],
    intervals: List[float],
    workflow_runs: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """Generate the HTML dashboard."""
    success_rate = calculate_success_rate(workflow_runs)
    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Metta Stability Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .tag-list {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .tag-item {{
            border-bottom: 1px solid #eee;
            padding: 15px 0;
        }}
        .tag-item:last-child {{
            border-bottom: none;
        }}
        .tag-name {{
            font-weight: bold;
            color: #007bff;
            font-size: 1.1em;
        }}
        .tag-date {{
            color: #666;
            font-size: 0.9em;
            margin-left: 10px;
        }}
        .tag-commit {{
            font-family: monospace;
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.85em;
        }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .timestamp {{
            color: #666;
            font-size: 0.8em;
            margin-top: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Metta Stability Dashboard</h1>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Total Stable Tags</div>
                <div class="metric-value">{len(tags)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Latest Stable Tag</div>
                <div class="metric-value" style="font-size: 1.5em;">{tags[0]['tag'] if tags else 'None'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Days Between Tags</div>
                <div class="metric-value">{avg_interval:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Success Rate</div>
                <div class="metric-value">{success_rate:.1f}%</div>
            </div>
        </div>
        
        <h2>Stable Tag History</h2>
        <div class="tag-list">
"""
    
    for i, tag in enumerate(tags):
        interval_text = ""
        if i < len(intervals):
            interval_text = f' <span style="color: #888;">(+{intervals[i]:.1f} days)</span>'
        
        html_content += f"""
            <div class="tag-item">
                <div>
                    <span class="tag-name">{tag['tag']}</span>
                    <span class="tag-date">{tag['timestamp']}</span>
                    {interval_text}
                </div>
                <div style="margin-top: 5px;">
                    <span class="tag-commit">{tag['commit'][:8]}</span>
                    <span style="margin-left: 10px; color: #666;">{tag['subject']}</span>
                </div>
            </div>
"""
    
    html_content += f"""
        </div>
        
        <div class="timestamp">
            Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
</body>
</html>
"""
    
    output_file = output_dir / "stability-dashboard.html"
    output_file.write_text(html_content)
    print(f"Dashboard generated: {output_file}")


def generate_dashboard_json(
    tags: List[Dict[str, Any]],
    intervals: List[float],
    workflow_runs: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """Generate JSON data for the dashboard."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_tags": len(tags),
            "latest_tag": tags[0] if tags else None,
            "average_interval_days": sum(intervals) / len(intervals) if intervals else 0,
            "test_success_rate": calculate_success_rate(workflow_runs)
        },
        "tags": tags,
        "intervals": intervals,
        "recent_workflow_runs": workflow_runs[:20]  # Last 20 runs
    }
    
    output_file = output_dir / "stability-data.json"
    output_file.write_text(json.dumps(data, indent=2))
    print(f"JSON data generated: {output_file}")


@click.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="stability-dashboard",
    help="Output directory for dashboard files"
)
@click.option(
    "--include-workflow-runs",
    is_flag=True,
    help="Include GitHub workflow run data (requires gh CLI)"
)
def main(output_dir: Path, include_workflow_runs: bool):
    """Generate a comprehensive stability dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting stable tags...")
    tags = get_stable_tags()
    
    if not tags:
        print("No stable tags found. Creating empty dashboard.")
        tags = []
    
    print(f"Found {len(tags)} stable tags")
    
    # Calculate intervals
    intervals = calculate_tag_intervals(tags)
    
    # Get workflow runs if requested
    workflow_runs = []
    if include_workflow_runs:
        print("Collecting workflow run data...")
        workflow_runs = get_workflow_runs()
        print(f"Found {len(workflow_runs)} workflow runs")
    
    # Generate outputs
    print("Generating dashboard...")
    generate_dashboard_html(tags, intervals, workflow_runs, output_dir)
    generate_dashboard_json(tags, intervals, workflow_runs, output_dir)
    
    print("Dashboard generation complete!")


if __name__ == "__main__":
    main()