import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from metta.app import config, dashboard_routes, stats_routes

app = fastapi.FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: add auth middleware

app.include_router(dashboard_routes.router)
app.include_router(stats_routes.router)

if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
