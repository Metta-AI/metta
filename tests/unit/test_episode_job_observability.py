import json
import uuid
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor

from metta.sim import single_episode_runner
from mettagrid import MettaGridConfig


class InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for testing."""

    def __init__(self) -> None:
        self._spans: list = []

    def export(self, spans) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list:
        return self._spans

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_run_episode_creates_observability_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(__name__)
    carrier: dict[str, str] = {}
    with tracer.start_as_current_span("parent-span") as parent_span:
        propagate.inject(carrier)
        parent_context = parent_span.get_span_context()

    env = MettaGridConfig.EmptyRoom(num_agents=1).model_dump()
    job_id = uuid.uuid4()
    job_payload = {
        "policy_uris": ["file://policy"],
        "assignments": [0],
        "env": env,
        "results_uri": "s3://results",
        "replay_uri": "s3://replay",
        "seed": 123,
        "max_action_time_ms": 1000,
        "episode_tags": {},
        "trace_context": carrier,
    }

    class StubStatsClient:
        def get_job(self, _job_id: uuid.UUID) -> SimpleNamespace:
            return SimpleNamespace(job=job_payload)

        def update_job(self, *_args: object, **_kwargs: object) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setenv("MACHINE_TOKEN", "token")
    monkeypatch.setenv("STATS_SERVER_URI", "http://stats")
    monkeypatch.setattr(single_episode_runner.StatsClient, "create", staticmethod(lambda _uri: StubStatsClient()))
    monkeypatch.setattr(single_episode_runner.observatory_auth_config, "save_token", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        single_episode_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(single_episode_runner, "copy_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        single_episode_runner,
        "read",
        lambda *_args, **_kwargs: json.dumps(
            {
                "rewards": [0.0],
                "action_timeouts": [0],
                "stats": {"game": {}, "agent": []},
                "steps": 1,
            }
        ),
    )
    episode_id = uuid.uuid4()
    monkeypatch.setattr(
        single_episode_runner,
        "write_single_episode_to_observatory",
        lambda **_kwargs: episode_id,
    )

    single_episode_runner.run_episode(job_id)

    spans = exporter.get_finished_spans()
    run_spans = [span for span in spans if span.name == "tournament.job.run"]
    assert run_spans, "Expected tournament.job.run span"
    run_span = run_spans[0]
    assert run_span.attributes["job.id"] == str(job_id)
    assert run_span.attributes["job.outcome"] == "success"
    assert run_span.attributes["episode.id"] == str(episode_id)
    assert run_span.parent is not None
    assert run_span.parent.span_id == parent_context.span_id

    step_names = {span.name for span in spans}
    assert "tournament.job.step.run_simulation" in step_names
    assert "tournament.job.step.upload_results" in step_names
    assert "tournament.job.step.write_episode" in step_names
    assert "tournament.job.step.update_job" in step_names
