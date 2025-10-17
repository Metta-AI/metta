"""Regression coverage for the AdA-inspired evaluation suite."""

from experiments.evals.ada import make_ada_eval_suite


def test_make_ada_eval_suite_contains_expected_scenarios() -> None:
    suite = make_ada_eval_suite()
    names = {sim.name for sim in suite}

    assert names == {"resource_chain", "weapons_lab", "cooperative_delivery"}

    for sim in suite:
        game = sim.env.game
        assert "altar" in game.objects

        if sim.name == "weapons_lab":
            assert game.actions.attack.enabled
            assert game.actions.attack.consumed_resources.get("laser") == 1

        if sim.name == "cooperative_delivery":
            assert "score_chest" in game.objects
