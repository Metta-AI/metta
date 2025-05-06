from typing import Literal

from metta.util.config import Config


class AnalyzerConfig(Config):
    # Output options
    num_output_policies: int | Literal["all"] = 20
    view_type: Literal["latest", "all"] = "latest"
    metric: str = "reward"
    output_path: str

    # Input database
    eval_db_uri: str

    # Filtering options
    suite: str | None = None
    policy_uri: str | None = None
