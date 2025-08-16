# Enable the new OpenAPI parser for better performance - must be set before imports
import os

os.environ["FASTMCP_EXPERIMENTAL_ENABLE_NEW_OPENAPI_PARSER"] = "true"

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastmcp import FastMCP

from metta.app_backend.config import host, port, stats_db_uri
from metta.app_backend.leaderboard_updater import LeaderboardUpdater
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app

if __name__ == "__main__":
    stats_repo = MettaRepo(stats_db_uri)
    app = create_app(stats_repo)

    # Create MCP after all routes are registered
    mcp = FastMCP.from_fastapi(app=app, name="metta-observatory")
    mcp_app = mcp.http_app(path="/")

    @asynccontextmanager
    async def combined_lifespan(app_):
        async with mcp_app.lifespan(app_):
            yield

    # Attach the MCP lifespan to the main app
    app.router.lifespan_context = combined_lifespan
    app.mount("/mcp", mcp_app)

    leaderboard_updater = LeaderboardUpdater(stats_repo)

    # Start the updater in an async context
    async def main():
        await leaderboard_updater.start()
        # Run uvicorn in a way that doesn't block
        config = uvicorn.Config(app, host=host, port=port)
        server = uvicorn.Server(config)

        print(f"🚀 Starting Metta Observatory server with MCP at http://{host}:{port}/mcp")
        await server.serve()

    asyncio.run(main())
