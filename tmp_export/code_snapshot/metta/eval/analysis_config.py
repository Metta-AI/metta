from mettagrid.base_config import Config


class AnalysisConfig(Config):
    # Policy URI to analyze
    policy_uri: str | None = None

    # Metrics to analyze
    # Supports globs, e.g. *.reward
    metrics: list[str] = ["*"]

    # Input database
    eval_db_uri: str

    # Filtering options
    sim_name: str | None = None

    # Output configuration (add these)
    output_path: str | None = None
    num_output_policies: str | int | None = None
