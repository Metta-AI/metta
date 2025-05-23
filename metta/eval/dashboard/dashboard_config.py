from typing import Literal

from metta.util.config import Config


class DashboardConfig(Config):
    __init__ = Config.__init__
    # Output options
    num_output_policies: int | Literal["all"] = 20
    metric: str = "reward"
    output_path: str = "/tmp/dashboard.html"
    # Input database
    eval_db_uri: str
    # Filtering options
    suite: str | None = None
