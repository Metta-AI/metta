from types import SimpleNamespace

import torch

from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.rl.training.evaluator import Evaluator, EvaluatorConfig


def test_evaluator_logs_replays_when_available(monkeypatch):
    config = EvaluatorConfig(evaluate_local=True, evaluate_remote=False, simulations=[])
    evaluator = Evaluator(
        config=config,
        device=torch.device("cpu"),
        system_cfg=SimpleNamespace(vectorization="serial"),
        stats_client=None,
    )

    fake_results = EvalResults(
        scores=EvalRewardSummary(),
        replay_urls={"eval/test": ["https://example.com/replay.json"]},
    )

    monkeypatch.setattr(Evaluator, "_build_simulations", lambda self, curriculum: [])
    monkeypatch.setattr(Evaluator, "_evaluate_local", lambda self, **kwargs: fake_results)

    logged_payloads = []

    def fake_upload(replay_urls, agent_step, epoch, wandb_run, step_metric_key, epoch_metric_key):
        logged_payloads.append(
            {
                "replay_urls": replay_urls,
                "agent_step": agent_step,
                "epoch": epoch,
                "wandb_run": wandb_run,
                "step_metric_key": step_metric_key,
                "epoch_metric_key": epoch_metric_key,
            }
        )

    monkeypatch.setattr("metta.rl.training.evaluator.upload_replay_html", fake_upload)

    evaluator._context = SimpleNamespace(stats_reporter=SimpleNamespace(wandb_run="sentinel"), agent_step=12)

    scores = evaluator.evaluate(policy_uri="file://policy", curriculum=object(), epoch=5, agent_step=12)

    assert isinstance(scores, EvalRewardSummary)
    assert logged_payloads, "Expected replay upload to be triggered"
    assert logged_payloads[0]["replay_urls"] == fake_results.replay_urls
    assert logged_payloads[0]["wandb_run"] == "sentinel"
