import os

from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from metta.app_backend.config import stats_db_uri
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app
from metta.common.util.stats_client_cfg import get_machine_token

os.environ["FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"] = "true"

# Set DEBUG_USER_EMAIL early for MCP stdio mode authentication
backend_url = os.environ.get("METTA_MCP_BACKEND_URL", "http://localhost:8000")
if backend_url != "http://localhost:8000":
    # For production backend, set debug user email to bypass authentication
    os.environ["DEBUG_USER_EMAIL"] = "zachary@stem.ai"

    # Also set production database URI for SQL routes if provided
    production_db_uri = os.environ.get("METTA_MCP_STATS_DB_URI", stats_db_uri)
    if production_db_uri:
        pass
    else:
        print("Warning: PRODUCTION_STATS_DB_URI not set, will use local database")


machine_token = get_machine_token(backend_url)
stats_repo = MettaRepo(stats_db_uri)
app = create_app(stats_repo)


# Create MCP from FastAPI app
mcp = FastMCP.from_fastapi(
    app=app,
    name="metta-observatory",
)

mcp.add_middleware(ErrorHandlingMiddleware())  # Handle errors first
mcp.add_middleware(RateLimitingMiddleware(max_requests_per_second=50))
mcp.add_middleware(TimingMiddleware())  # Time actual execution
mcp.add_middleware(LoggingMiddleware())  # Log everything


@mcp.tool()
async def get_auth_token() -> dict[str, str]:
    """Get the current user's email."""
    return {"X-Auth-Token": machine_token} if machine_token else {}


def main():
    """Run the MCP server in stdio mode."""
    mcp.run()


if __name__ == "__main__":
    main()
