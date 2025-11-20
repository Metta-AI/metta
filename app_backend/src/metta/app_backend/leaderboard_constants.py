"""Shared constants used by leaderboard components."""

LEADERBOARD_SIM_VERSION = "v0.1"
# Policy version tags
COGAMES_SUBMITTED_PV_KEY = "cogames-submitted"
LEADERBOARD_JOB_ID_PV_KEY = f"leaderboard-eval-remote-job-id-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_EVAL_DONE_PV_KEY = f"leaderboard-evals-done-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_ATTEMPTS_PV_KEY = f"leaderboard-attempts-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_EVAL_CANCELED_VALUE = "canceled"
LEADERBOARD_EVAL_DONE_VALUE = "true"

# Episode tags
LEADERBOARD_SIM_NAME_EPISODE_KEY = f"leaderboard-name-{LEADERBOARD_SIM_VERSION}"
