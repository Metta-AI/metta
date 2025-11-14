from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.sweep import SweepTool

# NOTE: other arena_basic_easy_shaped recipes are in recipes.prod


def sweep(sweep_name: str) -> SweepTool:
    """
    Prototypical sweep function.
    In your own recipe, you likely only every need this. You can override other SweepTool parameters in the CLI.

    Example usage:

        `uv run ./tools/run.py recipes.experiment.arena_basic_easy_shaped.sweep \
            sweep_name="ak.baes.10081528" -- gpus=4 nodes=2`


    We recommend running using local_test=True before running the sweep on the remote:
        `uv run ./tools/run.py recipes.prod.arena_basic_easy_shaped.sweep \
            sweep_name="ak.baes.10081528.local_test" -- local_test=True`

    This will run a quick local sweep and allow you to catch configuration bugs
    (NB: Unless those bugs are related to batch_size, minibatch_size, or hardware config).
    If this runs smoothly, you must launch the sweep on a remote sandbox
    (otherwise sweep progress will halt when you close your computer).

    Running on the remote:
        1 - Start a sweep controller sandbox: `./devops/skypilot/sandbox.py --sweep-controller`, and ssh into it.
        2 - Clean git pollution: `git clean -df && git stash`
        3 - Ensure your sky credentials are present: `sky status` -- if not, follow the instructions on screen.
        4 - Install tmux on the sandbox `apt install tmux`
        5 - Launch tmux session: `tmux new -s sweep`
        6 - Launch the sweep:
            `uv run ./tools/run.py recipes.prod.arena_basic_easy_shaped.sweep \
                sweep_name="ak.baes.10081528" -- gpus=4 nodes=2`
        7 - Detach when you want: CTRL+B then d
        8 - Attach to look at status/output: `tmux attach -t sweep_configs`

    Please tag Axel (akerbec@softmax.ai) on any bug report.
    """

    # Common parameters are accessible via SP (SweepParameters).
    parameters = [
        SP.LEARNING_RATE,
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.ADAM_EPS,
        SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=7.5e8,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.arena_basic_easy_shaped",
        train_entrypoint="train",
        # NB: You MUST use a specific sweep eval suite, different than those in training.
        # Besides this being a recommended practice, using the same eval suite in both
        # training and scoring will lead to key conflicts that will lock the sweep.
        eval_entrypoint="evaluate_in_sweep",
        # Typically, "evaluator/eval_{suite}/score"
        objective="evaluator/eval_sweep/score",
        # TODO: axel: needs fixing
        parameters=parameters,
        max_trials=80,
        # Default value is 1. We don't recommend going higher than 4.
        # The faster each individual trial, the lower you should set this number.
        num_parallel_trials=4,
    )
