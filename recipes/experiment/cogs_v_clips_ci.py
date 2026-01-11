"""CI/stable job registrations for cogs_v_clips."""

from devops.stable.registry import ci_job, stable_job
from devops.stable.runner import AcceptanceCriterion
from recipes.experiment import cogs_v_clips

train_ci = ci_job(timeout_s=240)(cogs_v_clips.train_ci)
play_ci = ci_job(timeout_s=120)(cogs_v_clips.play_ci)

train_200ep = stable_job(
    remote_gpus=1,
    remote_nodes=1,
    timeout_s=43200,
    acceptance=[AcceptanceCriterion(metric="overview/sps", threshold=13000)],
)(cogs_v_clips.train_200ep)

train_2b = stable_job(
    remote_gpus=4,
    remote_nodes=4,
    timeout_s=172800,
    acceptance=[AcceptanceCriterion(metric="overview/sps", threshold=80000)],
)(cogs_v_clips.train_2b)
