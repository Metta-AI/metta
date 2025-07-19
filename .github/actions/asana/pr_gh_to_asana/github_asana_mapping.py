from typing import Generator

import requests


class GithubAsanaMapping:
    def __init__(
        self,
        github_logins: set[str],
        roster_project_id: str,
        gh_login_field_id: str,
        asana_email_field_id: str,
        asana_token: str,
    ):
        self.github_login_to_asana_email = {}
        mapping = {}
        for task in self._get_task_custom_fields_from_project(roster_project_id, asana_token):
            custom_fields = task.get("custom_fields") or []
            gh_login = None
            asana_email = None
            for field in custom_fields:
                if isinstance(field, dict):
                    if field.get("gid") == gh_login_field_id:
                        gh_login = value.strip() if (value := field.get("text_value")) else None
                        if gh_login not in github_logins:
                            break
                    if field.get("gid") == asana_email_field_id:
                        asana_email = field.get("text_value")
                        if asana_email and asana_email.strip() == "mh.next@gmail.com, mhollander@stem.ai":
                            asana_email = "mh.next@gmail.com"
                    if gh_login and asana_email:
                        mapping[gh_login] = asana_email
                        break
            if len(mapping) == len(github_logins):
                break
        print(f"github_login_to_asana_email: {mapping}")
        self.github_login_to_asana_email = mapping

    def _get_task_custom_fields_from_project(
        self,
        project_id: str,
        asana_token: str,
    ) -> Generator[dict[str, str], None, None]:
        url = f"https://app.asana.com/api/1.0/projects/{project_id}/tasks"
        headers = {
            "Authorization": f"Bearer {asana_token}",
            "Content-Type": "application/json",
        }
        params = {
            "opt_fields": "custom_fields",
            "limit": 100,
        }
        while True:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Asana API Error (_get_task_custom_fields_from_project): {response.status_code} - {response.text}"
                )
            data = response.json()
            tasks = data["data"]
            for task in tasks:
                yield task
            if not data.get("next_page"):
                break
            params["offset"] = data["next_page"]["offset"]
