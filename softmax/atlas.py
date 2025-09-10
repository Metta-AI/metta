from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Softmax Codex")

# Store roles with their prompts and required context
roles = {
    "peer_feedback_coach": {
        "name": "360 Feedback Coach",
        "description": "Experienced coach conducting peer feedback interviews",
        "required_context": ["employee_name", "employee_roles", "projects_worked_together"],
        "prompt": """This is the Softmax Peer Feedback prompt.
You are an experienced coach gathering 360 feedback...""",
    },
    "self_reflection_coach": {
        "name": "Self-Reflection Coach",
        "description": "Guide for employee self-assessment",
        "required_context": ["employee_roles", "peer_feedback_transcripts"],
        "prompt": """Self-reflection prompt text...""",
    },
}


@mcp.tool("list_roles")
def list_roles() -> dict:
    """List all available roles with descriptions"""
    return {
        "roles": [
            {"id": role_id, "name": role_data["name"], "description": role_data["description"]}
            for role_id, role_data in roles.items()
        ]
    }


@mcp.tool("get_role_requirements")
def get_role_requirements(role_id: str) -> dict:
    """Get details about what context a role needs"""
    if role_id not in roles:
        return {"error": f"Role '{role_id}' not found"}

    role = roles[role_id]
    return {
        "role_id": role_id,
        "name": role["name"],
        "required_context": role["required_context"],
        "description": role["description"],
    }


@mcp.tool("start_role")
def start_role(role_id: str, context: dict) -> str:
    """Initialize a role with the provided context"""
    if role_id not in roles:
        return f"Error: Role '{role_id}' not found"

    role = roles[role_id]

    # Validate required context
    missing = [field for field in role["required_context"] if field not in context]
    if missing:
        return f"Error: Missing required context fields: {missing}"

    # Format the prompt with context
    context_block = "\n".join([f"{k}: {v}" for k, v in context.items()])

    return f"""ROLE INITIALIZATION: {role["name"]}

CONTEXT PROVIDED:
{context_block}

ROLE PROMPT:
{role["prompt"]}

You should now embody this role and begin the interaction."""
