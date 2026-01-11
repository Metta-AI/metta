"""CI/stable job registrations for arena_basic_easy_shaped."""

from devops.stable.registry import ci_job, stable_job
from devops.stable.runner import AcceptanceCriterion
from recipes.prod import arena_basic_easy_shaped

play_ci = ci_job(timeout_s=120)(arena_basic_easy_shaped.play_ci)

train_100m = stable_job(
    remote_gpus=1,
    remote_nodes=1,
    timeout_s=7200,
    acceptance=[
        AcceptanceCriterion(metric="overview/sps", threshold=23000),
        AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
    ],
)(arena_basic_easy_shaped.train_100m)

train_2b = stable_job(
    remote_gpus=4,
    remote_nodes=4,
    timeout_s=172800,
    acceptance=[
        AcceptanceCriterion(metric="overview/sps", threshold=80000),
        AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
    ],
)(arena_basic_easy_shaped.train_2b)
