import json
from typing import Sequence

from pydantic import Field

from metta.common.tool.tool import ToolResult, ToolWithResult
from metta.sim.runner import SimulationRunConfig
from metta.tools.eval import EvaluatePolicyVersionTool
from metta.tools.utils.auto_config import auto_replay_dir


class ExecuteRemoteEvalTool(ToolWithResult):
    simulations: Sequence[SimulationRunConfig]  # list of simulations to run
    policy_version_id: str  # policy uri to evaluate
    eval_task_id: str
    replay_dir: str = Field(default_factory=auto_replay_dir)
    result_file_path: str  # path to the file where the results will be written

    group: str | None = None  # Separate group parameter like in train.py

    stats_server_uri: str | None = None  # If set, send stats to this http server
    push_metrics_to_wandb: bool = True

    def run_job(self) -> ToolResult:
        # Will error if stats_server_uri does not exist or we are not not authenticated with it
        if self.stats_server_uri is None:
            raise ValueError("stats_server_uri is required")

        eval_tool = EvaluatePolicyVersionTool(
            simulations=self.simulations,
            policy_version_id=self.policy_version_id,
            replay_dir=self.replay_dir,
            group=self.group,
            stats_server_uri=self.stats_server_uri,
        )
        try:
            eval_tool.invoke({})
            return ToolResult(result="success")
        except Exception as e:
            return ToolResult(result="failure", error=str(e))


# Used by eval_task_worker.py
def eval(
    policy_version_id: str,
    task_data_path: str,
    result_file_path: str,
    eval_task_id: str,
) -> ExecuteRemoteEvalTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    with open(task_data_path, "rb") as f:
        simulations_json = f.read()
    sim_json_array = json.loads(simulations_json)["simulations"]
    simulations = [SimulationRunConfig.model_validate(sim) for sim in sim_json_array]

    return ExecuteRemoteEvalTool(
        simulations=simulations,
        policy_version_id=policy_version_id,
        result_file_path=result_file_path,
        eval_task_id=eval_task_id,
    )
