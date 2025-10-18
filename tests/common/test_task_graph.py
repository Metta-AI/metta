import pytest

from metta.common import TaskGraph


class TestTaskGraph:
    def test_add_tasks_and_dependencies(self) -> None:
        graph = TaskGraph[str]()

        graph.add_task("fetch_data", payload="download")
        graph.add_task("train", payload="train", dependencies=["fetch_data", "prepare"])
        graph.add_task("prepare", payload="clean", dependencies=["fetch_data"])

        assert graph.dependencies_of("train") == {"fetch_data", "prepare"}
        assert graph.dependencies_of("prepare") == {"fetch_data"}
        assert graph.dependents_of("fetch_data") == {"prepare", "train"}
        assert graph.get_payload("prepare") == "clean"

    def test_topological_order(self) -> None:
        graph = TaskGraph[None]()
        graph.add_task("lint")
        graph.add_task("unit_tests", dependencies=["lint"])
        graph.add_task("integration_tests", dependencies=["unit_tests"])
        graph.add_task("deploy", dependencies=["integration_tests", "lint"])

        order = graph.topological_order()

        assert order.index("lint") < order.index("unit_tests")
        assert order.index("unit_tests") < order.index("integration_tests")
        assert order.index("integration_tests") < order.index("deploy")

    def test_topological_order_cycle_detection(self) -> None:
        graph = TaskGraph[None]()
        graph.add_task("a", dependencies=["b"])
        graph.add_task("b", dependencies=["a"])

        with pytest.raises(ValueError, match="cycle"):
            graph.topological_order()

    def test_ready_tasks(self) -> None:
        graph = TaskGraph[None]()
        graph.add_task("lint")
        graph.add_task("test", dependencies=["lint"])
        graph.add_task("package", dependencies=["lint", "test"])

        assert graph.ready_tasks() == ["lint"]
        assert graph.ready_tasks({"lint"}) == ["test"]
        assert graph.ready_tasks({"lint", "test"}) == ["package"]
