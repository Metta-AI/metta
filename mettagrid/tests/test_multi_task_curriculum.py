import pytest
from omegaconf import OmegaConf

from mettagrid.curriculum import MultiTaskCurriculum, RandomCurriculum, SingleTaskCurriculum


class DummyCurriculum(SingleTaskCurriculum):
    def __init__(self, num_agents: int):
        cfg = OmegaConf.create({"game": {"num_agents": num_agents}})
        super().__init__("dummy", cfg)


class TestMultiTaskCurriculum:
    def test_random_curriculum_agent_mismatch(self, monkeypatch):
        def fake_from_config(path, env_overrides=None):
            if path == "a":
                return DummyCurriculum(1)
            return DummyCurriculum(2)

        monkeypatch.setattr(MultiTaskCurriculum, "from_config_path", staticmethod(fake_from_config))

        with pytest.raises(AssertionError):
            RandomCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))
