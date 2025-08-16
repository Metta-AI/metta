# Enable the new OpenAPI parser for better performance - must be set before imports
import os

os.environ["FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"] = "true"

from fastapi.security import HTTPBearer
from fastmcp import FastMCP

from metta.app_backend.config import stats_db_uri
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app
from metta.common.util.stats_client_cfg import get_machine_token

security = HTTPBearer()


backend_url = os.environ.get("METTA_MPC_BACKEND_URL") or "http://localhost:8000"
machine_token = get_machine_token(backend_url)


def main():
    """Run the MCP server in stdio mode."""
    stats_repo = MettaRepo(stats_db_uri)
    app = create_app(stats_repo)

    # Create MCP from FastAPI app
    mcp = FastMCP.from_fastapi(
        app=app,
        name="metta-observatory",
        httpx_client_kwargs={
            "headers": {
                "X-Auth-Token": machine_token,
            }
        },
    )

    print("🚀 Starting Metta Observatory MCP server in stdio mode", file=sys.stderr)

    # Run in stdio mode
    mcp.run()


if __name__ == "__main__":
    import sys

    main()
