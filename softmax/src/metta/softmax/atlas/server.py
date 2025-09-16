from typing import Any

from mcp.server.fastmcp import FastMCP

from metta.softmax.asana.app_authenticate import get_asana_client

SOFTMAX_ATLAS_NAME = "Softmax Atlas"
SOFTMAX_ATLAS_ASANA_APP = "atlas"
ASANA_ATLAS_PROJECT_GID = "1211363606493626"  # Atlas Prompts

mcp = FastMCP(SOFTMAX_ATLAS_NAME)


def _fetch_roles_from_asana() -> list[dict[str, Any]]:
    client = get_asana_client(SOFTMAX_ATLAS_ASANA_APP)
    roles: list[dict[str, Any]] = []
    fields = "name,notes,permalink_url"
    params: dict[str, Any] = {"opt_fields": fields, "limit": 100}
    next_page: dict[str, Any] | None = None
    while True:
        if next_page and (offset := next_page.get("offset")):
            params["offset"] = offset
        resp = client.get(f"/projects/{ASANA_ATLAS_PROJECT_GID}/tasks", params=params)
        resp.raise_for_status()
        payload = resp.json()
        roles.extend(
            [
                {
                    "id": task.get("gid"),
                    "name": task.get("name") or f"Untitled {task.get('gid')}",
                    "prompt": task.get("notes") or "",
                }
                for task in payload.get("data", [])
            ]
        )

        next_page = payload.get("next_page")
        if not next_page:
            break

    return roles


@mcp.tool("list_roles")
def list_roles() -> dict:
    return {"roles": _fetch_roles_from_asana()}


@mcp.tool("start_role")
def start_role(role_id: str) -> str:
    roles = _fetch_roles_from_asana()
    role = next((r for r in roles if r["id"] == role_id), None)

    if role is None:
        return f"Error: Role '{role_id}' not found"

    return f"""ROLE INITIALIZATION: {role["name"]}

ROLE PROMPT:
{role["prompt"]}

You should now embody this role and begin the interaction."""
