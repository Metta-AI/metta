from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import requests

DASHBOARD_DIR = Path(__file__).parent / "dashboards"


def load_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Environment variable {var} is required")
    return value


def push_dashboard(file_path: Path, site: str, api_key: str, app_key: str) -> None:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    url = f"https://{site.rstrip('/')}/api/v1/dashboard"
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "DD-API-KEY": api_key,
            "DD-APPLICATION-KEY": app_key,
        },
        data=json.dumps(payload),
        timeout=30,
    )
    if response.status_code >= 300:
        raise RuntimeError(f"Failed to push {file_path.name}: {response.status_code} {response.text}")
    data = response.json()
    dashboard_url = data.get("url") or data.get("public_url") or "N/A"
    print(f"Uploaded {file_path.name} -> {dashboard_url}")


def list_dashboard_files(selected: Iterable[str] | None = None) -> List[Path]:
    files = []
    for name in ("infra_summary.json", "infra_detailed.json"):
        path = DASHBOARD_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Dashboard file missing: {path}")
        files.append(path)
    if selected:
        mapping = {path.name: path for path in files}
        files = [mapping[name] for name in selected if name in mapping]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Datadog dashboards to the API.")
    parser.add_argument(
        "--dashboard",
        action="append",
        help="Specific dashboard file name(s) to upload (default uploads both summary and detailed).",
    )
    args = parser.parse_args()

    site = load_env("DD_SITE")
    api_key = load_env("DD_API_KEY")
    app_key = load_env("DD_APP_KEY")

    files = list_dashboard_files(args.dashboard)
    for file_path in files:
        push_dashboard(file_path, site=site, api_key=api_key, app_key=app_key)


if __name__ == "__main__":
    main()
