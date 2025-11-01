import math

import pytest

from metta.common.task_graph import LearningTaskGraph


class TestLearningTaskGraph:
    def build_graph(self) -> LearningTaskGraph:
        graph = LearningTaskGraph(base_learning_rate=0.2, decay_rate=0.05, prereq_threshold=0.4)
        graph.add_task("navigation", difficulty=0.4)
        graph.add_task("maze", difficulty=0.6, dependencies=["navigation"])
        graph.add_task("boss", difficulty=0.9, dependencies=["maze"])
        graph.add_task("sidequest", difficulty=0.3)
        return graph

    def test_prerequisite_gate(self):
        graph = self.build_graph()

        # prerequisite competence too low -> no learning
        learned = graph.record_experience("maze")
        assert not learned
        assert graph.get_competence("maze") == 0.0

        # Train prerequisite until threshold is cleared
        for _ in range(10):
            graph.record_experience("navigation")

        assert graph.get_competence("navigation") > graph.prereq_threshold

        learned = graph.record_experience("maze")
        assert learned
        assert graph.get_competence("maze") > 0.0

    def test_decay_applies_to_other_tasks(self):
        graph = self.build_graph()
        graph.record_experience("sidequest")
        base = graph.get_competence("sidequest")
        assert base > 0.0

        graph.record_experience("navigation")
        decayed = graph.get_competence("sidequest")
        assert decayed < base

    def test_simulation_resets_state(self):
        graph = self.build_graph()
        result = graph.simulate_schedule(["navigation", "maze", "boss"], experience=2.0)

        assert len(result.steps) == 3
        assert pytest.approx(sum(result.final_competence.values()), rel=1e-6) == result.total_competence

        # After simulation the graph should be back to its original competences (zero)
        for task_id in ["navigation", "maze", "boss", "sidequest"]:
            assert graph.get_competence(task_id) == 0.0

    def test_optimal_order_and_regret(self):
        graph = self.build_graph()
        optimal = graph.compute_optimal_learning_order()
        assert optimal[0] in {"navigation", "sidequest"}
        assert optimal[-1] == "boss"

        # Actual schedule flips the order leading to regret
        schedule = ["sidequest", "sidequest", "sidequest", "navigation"]
        regret, optimal_result, actual_result = graph.compute_regret(schedule)

        assert regret >= 0
        assert math.isclose(optimal_result.total_competence, sum(optimal_result.final_competence.values()))
        assert math.isclose(actual_result.total_competence, sum(actual_result.final_competence.values()))

        # Regret should be positive because the optimal order mixes navigation earlier
        assert regret > 0
