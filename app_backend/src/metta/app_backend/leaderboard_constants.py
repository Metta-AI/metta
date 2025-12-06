"""Shared constants used by leaderboard components."""

LEADERBOARD_SIM_VERSION = "v0.2"

# Policy version tags
COGAMES_SUBMITTED_PV_KEY = "cogames-submitted"
LEADERBOARD_JOB_ID_PV_KEY = f"leaderboard-eval-remote-job-id-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_EVAL_DONE_PV_KEY = f"leaderboard-evals-done-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_ATTEMPTS_PV_KEY = f"leaderboard-attempts-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_EVAL_CANCELED_VALUE = "canceled"
LEADERBOARD_EVAL_DONE_VALUE = "true"

# Episode tags
LEADERBOARD_SIM_NAME_EPISODE_KEY = f"leaderboard-name-{LEADERBOARD_SIM_VERSION}"
LEADERBOARD_SCENARIO_KEY = "leaderboard-scenario-key"
LEADERBOARD_SCENARIO_KIND_KEY = "leaderboard-scenario-kind"
LEADERBOARD_CANDIDATE_COUNT_KEY = "leaderboard-candidate-count"
LEADERBOARD_THINKY_COUNT_KEY = "leaderboard-thinky-count"
LEADERBOARD_LADYBUG_COUNT_KEY = "leaderboard-ladybug-count"


THINKY_UUID = "674fc022-5f1f-41e5-ab9e-551fa329b723"
LADYBUG_UUID = "5a491d05-7fb7-41a0-a250-fe476999edcd"

# Hardcoded replacement baseline (average reward of thinky/ladybug in c0 scenarios)
# This avoids needing to run c0 scenarios for every policy evaluation
REPLACEMENT_BASELINE_MEAN = 4.9
