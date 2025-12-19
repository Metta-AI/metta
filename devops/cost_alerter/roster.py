import json
from dataclasses import dataclass
from typing import Generator

import requests

from softmax.aws.secrets_manager import get_secretsmanager_secret

ROSTER_PROJECT_ID = "1209948553419016"


@dataclass
class RosterEntry:
    name: str
    email: str
    mentor_email: str | None
    cost_threshold_daily: float
    job_threshold: int


def _get_asana_token() -> str:
    token = get_secretsmanager_secret("asana/access-token", require_exists=False)
    if token:
        return token.strip()

    secret_str = get_secretsmanager_secret("asana/atlas_app", require_exists=False)
    if secret_str:
        secret_data = json.loads(secret_str)
        token = secret_data.get("token") or secret_data.get("ASANA_TOKEN")
        if token:
            return token

    raise ValueError("No Asana token found in asana/access-token or asana/atlas_app")


def _get_project_custom_field_ids(project_id: str, token: str) -> dict[str, str]:
    url = f"https://app.asana.com/api/1.0/projects/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"opt_fields": "custom_field_settings.custom_field.name,custom_field_settings.custom_field.gid"}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    field_map = {}
    for setting in response.json()["data"].get("custom_field_settings", []):
        cf = setting.get("custom_field", {})
        name = cf.get("name", "").lower().replace(" ", "_")
        gid = cf.get("gid")
        if name and gid:
            field_map[name] = gid

    return field_map


def _iter_project_tasks(project_id: str, token: str) -> Generator[dict, None, None]:
    url = f"https://app.asana.com/api/1.0/projects/{project_id}/tasks"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "opt_fields": "name,custom_fields,assignee.email",
        "limit": 100,
    }

    while True:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for task in data["data"]:
            yield task

        if not data.get("next_page"):
            break
        params["offset"] = data["next_page"]["offset"]


def _get_custom_field_value(task: dict, field_id: str) -> str | None:
    for field in task.get("custom_fields", []):
        if field.get("gid") == field_id:
            if field.get("text_value"):
                return field["text_value"]
            if field.get("display_value"):
                return field["display_value"]
            if field.get("number_value") is not None:
                return str(field["number_value"])
    return None


def get_roster(
    default_cost_threshold: float = 200.0,
    default_job_threshold: int = 5,
) -> list[RosterEntry]:
    token = _get_asana_token()
    field_ids = _get_project_custom_field_ids(ROSTER_PROJECT_ID, token)

    print(f"Found Asana custom fields: {list(field_ids.keys())}")

    email_field = field_ids.get("email") or field_ids.get("asana_email")
    mentor_field = field_ids.get("cost_monitor") or field_ids.get("mentor") or field_ids.get("cost_mentor")
    threshold_field = field_ids.get("cost_threshold") or field_ids.get("daily_cost_limit")
    job_threshold_field = field_ids.get("job_threshold") or field_ids.get("max_jobs")

    entries = []
    for task in _iter_project_tasks(ROSTER_PROJECT_ID, token):
        name = task.get("name", "")
        if not name or name.startswith("["):
            continue

        assignee = task.get("assignee") or {}
        email = assignee.get("email")
        if not email and email_field:
            email = _get_custom_field_value(task, email_field)

        if not email:
            continue

        mentor_email = None
        if mentor_field:
            mentor_email = _get_custom_field_value(task, mentor_field)

        cost_threshold = default_cost_threshold
        if threshold_field:
            val = _get_custom_field_value(task, threshold_field)
            if val:
                try:
                    cost_threshold = float(val)
                except ValueError:
                    pass

        job_threshold = default_job_threshold
        if job_threshold_field:
            val = _get_custom_field_value(task, job_threshold_field)
            if val:
                try:
                    job_threshold = int(float(val))
                except ValueError:
                    pass

        entries.append(
            RosterEntry(
                name=name,
                email=email,
                mentor_email=mentor_email,
                cost_threshold_daily=cost_threshold,
                job_threshold=job_threshold,
            )
        )

    print(f"Loaded {len(entries)} roster entries from Asana")
    return entries
