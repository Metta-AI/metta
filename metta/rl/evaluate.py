import logging
import typing
import uuid

import wandb

import metta.app_backend.clients.stats_client
import metta.app_backend.routes.eval_task_routes
import metta.common.util.collections
import metta.common.util.constants
import metta.common.wandb.context
import metta.rl.checkpoint_manager
import metta.sim.simulation_config
import metta.sim.utils

logger = logging.getLogger(__name__)


# Avoid circular import: evaluator.py → evaluate.py → evaluator.py
# EvaluatorConfig is only used as a type hint, never instantiated here
if typing.TYPE_CHECKING:
    import metta.rl.training.evaluator


def evaluate_policy_remote_with_checkpoint_manager(
    policy_uri: str,
    simulations: list[metta.sim.simulation_config.SimulationConfig],
    stats_epoch_id: uuid.UUID | None,
    stats_client: metta.app_backend.clients.stats_client.StatsClient | None,
    wandb_run: metta.common.wandb.context.WandbRun | None,
    evaluation_cfg: metta.rl.training.evaluator.EvaluatorConfig | None,
) -> metta.app_backend.routes.eval_task_routes.TaskResponse | None:
    """Create a remote evaluation task using a policy URI."""
    if not (wandb_run and stats_client and policy_uri):
        logger.warning("Remote evaluation requires wandb_run, stats_client, and policy_uri")
        return None

    # Normalize the policy URI
    normalized_uri = metta.rl.checkpoint_manager.CheckpointManager.normalize_uri(policy_uri)

    # Process policy registration using the new format
    stats_server_policy_id = metta.sim.utils.get_or_create_policy_ids(
        stats_client,
        [(normalized_uri, wandb_run.notes)],  # New format: (uri, description)
        stats_epoch_id,
    ).get(normalized_uri)

    if not stats_server_policy_id:
        logger.warning(f"Remote evaluation: failed to get or register policy ID for {normalized_uri}")
        return None

    # Create evaluation task
    task = stats_client.create_task(
        metta.app_backend.routes.eval_task_routes.TaskCreateRequest(
            policy_id=stats_server_policy_id,
            sim_suite=simulations[0].name,
            attributes={
                "git_hash": (evaluation_cfg and evaluation_cfg.git_hash),
                "simulations": [sim.model_dump() for sim in simulations],
            },
        )
    )

    logger.info(f"Policy evaluator: created task {task.id} for {normalized_uri} on {simulations[0].name}")

    return task


def upload_replay_html(
    replay_urls: dict[str, list[str]],
    agent_step: int,
    epoch: int,
    wandb_run: metta.common.wandb.context.WandbRun,
    step_metric_key: str | None = None,
    epoch_metric_key: str | None = None,
) -> None:
    """Upload organized replay HTML links to wandb."""
    # Create unified HTML with all replay links on a single line
    if replay_urls:
        # Group replays by base name
        replay_groups = {}

        for sim_name, urls in sorted(replay_urls.items()):
            if "training_task" in sim_name:
                # Training replays
                if "training" not in replay_groups:
                    replay_groups["training"] = []
                replay_groups["training"].extend(urls)
            else:
                # Evaluation replays - clean up the display name
                display_name = sim_name.replace("eval/", "")
                if display_name not in replay_groups:
                    replay_groups[display_name] = []
                replay_groups[display_name].extend(urls)

        # Build HTML with episode numbers
        links = []
        for name, urls in replay_groups.items():
            if len(urls) == 1:
                # Single episode - just show the name
                links.append(_form_mettascope_link(urls[0], name))
            else:
                # Multiple episodes - show name with numbered links
                episode_links = []
                for i, url in enumerate(urls, 1):
                    episode_links.append(_form_mettascope_link(url, str(i)))
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Log all links in a single HTML entry
        html_content = " | ".join(links)
        _upload_replay_html(html_content, agent_step, epoch, wandb_run, step_metric_key, epoch_metric_key)


def _form_mettascope_link(url: str, name: str) -> str:
    return f'<a href="{metta.common.util.constants.METTASCOPE_REPLAY_URL_PREFIX}{url}" target="_blank">{name}</a>'


def _upload_replay_html(
    html_content: str,
    agent_step: int,
    epoch: int,
    wandb_run: metta.common.wandb.context.WandbRun,
    step_metric_key: str | None = None,
    epoch_metric_key: str | None = None,
) -> None:
    payload: dict[str, typing.Any] = metta.common.util.collections.remove_none_keys(
        {"replays/all": wandb.Html(html_content), step_metric_key: agent_step, epoch_metric_key: epoch}
    )
    if step_metric_key or epoch_metric_key:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=epoch)
