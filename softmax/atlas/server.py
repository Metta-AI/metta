import time
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from softmax.asana.app_authenticate import get_authenticated_asana_client  # noqa: E402

SOFTMAX_ATLAS_NAME = "Softmax Atlas"
SOFTMAX_ATLAS_ASANA_APP = "atlas"
SOFTMAX_ATLAS_SERVER_PATH = __file__
SOFTMAX_ATLAS_ASANA_PROJECT = "Atlas Prompts"

mcp = FastMCP(SOFTMAX_ATLAS_NAME)

_roles_cache: dict[str, Any] = {
    "expires_at": 0.0,
    "roles_by_id": {},
    "project_gid": None,
}


def _truncate(text: str, length: int = 160) -> str:
    text = (text or "").strip()
    if len(text) <= length:
        return text
    return text[: max(0, length - 1)].rstrip() + "\u2026"


def _resolve_project_gid(client: httpx.Client, *, project_name: str) -> str:
    workspaces_resp = client.get("/workspaces")
    workspaces_resp.raise_for_status()
    for ws in workspaces_resp.json().get("data", []):
        params = {"workspace": ws["gid"], "archived": False, "limit": 100}
        projects_resp = client.get("/projects", params=params)
        projects_resp.raise_for_status()
        for proj in projects_resp.json().get("data", []):
            if proj.get("name") == project_name:
                return proj["gid"]
    raise Exception("Could not find GID for project named '%s'" % project_name)


def _fetch_roles_from_asana(client: httpx.Client, project_gid: str) -> dict[str, dict[str, Any]]:
    roles_by_id: dict[str, dict[str, Any]] = {}

    # Fetch tasks from the project; each task becomes a role
    fields = "name,notes,permalink_url"
    params: dict[str, Any] = {"opt_fields": fields, "limit": 100}
    next_page: dict[str, Any] | None = None

    while True:
        if next_page and (offset := next_page.get("offset")):
            params["offset"] = offset
        resp = client.get(f"/projects/{project_gid}/tasks", params=params)
        resp.raise_for_status()
        payload = resp.json()
        for task in payload.get("data", []):
            role_id = task.get("gid")
            name = task.get("name") or f"Untitled {role_id}"
            prompt = (task.get("notes") or "").strip()
            description = _truncate(prompt.splitlines()[0] if prompt else "")

            roles_by_id[role_id] = {
                "id": role_id,
                "name": name,
                "description": description,
                "prompt": prompt,
                "required_context": [],
                "permalink_url": task.get("permalink_url"),
            }

        next_page = payload.get("next_page")
        if not next_page:
            break

    return roles_by_id


def get_atlas_asana_client() -> httpx.Client:
    return get_authenticated_asana_client(SOFTMAX_ATLAS_ASANA_APP)


def _ensure_roles_loaded() -> None:
    now = time.time()
    if _roles_cache["expires_at"] > now and _roles_cache["roles_by_id"]:
        return

    client = get_atlas_asana_client()

    project_gid = _resolve_project_gid(client, project_name=SOFTMAX_ATLAS_ASANA_PROJECT)

    roles_by_id = _fetch_roles_from_asana(client, project_gid)

    _roles_cache["roles_by_id"] = roles_by_id
    _roles_cache["project_gid"] = project_gid
    _roles_cache["expires_at"] = now + 600


@mcp.tool("list_roles")
def list_roles() -> dict:
    """List all available roles with descriptions (from Asana)."""
    _ensure_roles_loaded()
    roles_by_id = _roles_cache["roles_by_id"]

    return {
        "roles": [
            {"id": role_id, "name": role_data["name"], "description": role_data["description"]}
            for role_id, role_data in roles_by_id.items()
        ]
    }


@mcp.tool("get_role_requirements")
def get_role_requirements(role_id: str) -> dict:
    """Get details about what context a role needs (currently none)."""
    _ensure_roles_loaded()
    roles_by_id = _roles_cache["roles_by_id"]

    if role_id not in roles_by_id:
        return {"error": f"Role '{role_id}' not found"}

    role = roles_by_id[role_id]
    return {
        "role_id": role_id,
        "name": role["name"],
        "required_context": role["required_context"],
        "description": role["description"],
    }


@mcp.tool("start_role")
def start_role(role_id: str, context: dict[str, Any]) -> str:
    """Initialize a role with the provided context."""
    _ensure_roles_loaded()
    roles_by_id = _roles_cache["roles_by_id"]

    if role_id not in roles_by_id:
        return f"Error: Role '{role_id}' not found"

    role = roles_by_id[role_id]

    # No required context for now
    context_block = "\n".join([f"{k}: {v}" for k, v in context.items()]) if context else "(none)"

    return f"""ROLE INITIALIZATION: {role["name"]}

CONTEXT PROVIDED:
{context_block}

ROLE PROMPT:
{role["prompt"]}

You should now embody this role and begin the interaction."""
