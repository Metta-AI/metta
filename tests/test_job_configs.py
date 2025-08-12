from pathlib import Path
import pytest
import hydra
from omegaconf import OmegaConf

def get_entrypoint_configs():
    """
    Dynamically discovers entrypoint configs by scanning the top-level of the 
    'configs' directory. Excludes known non-entrypoint files.
    """
    config_root = Path(__file__).parent.parent / "configs"
    
    # Files in the top-level directory that are not entrypoints
    exclude_files = {"common.yaml", "hydra.yaml", "defaults.yaml"}

    entrypoints = []
    for f in config_root.iterdir():
        if f.is_file() and f.suffix == ".yaml" and f.name not in exclude_files:
            entrypoints.append(f.name)
    
    return entrypoints

# This dictionary simulates the mandatory, dynamic parameters that a user
# would provide on the command line for each specific job. These configs are
# designed to fail without this information, as it represents runtime-specific
# data like database paths or experiment tags that should not be checked into git.
# By providing dummy values here, we can test the integrity of the config composition
# as if it were a real run.
# The `+` prefix is required by Hydra to add a key to a "structured" config,
# which prevents accidental addition of misspelled keys.
REQUIRED_RUNTIME_OVERRIDES = {
    # analyze_job requires the location of the evaluation database and the policy to analyze.
    "analyze_job.yaml": ["+eval_db_uri=???", "+policy_uri=???", "run=test"],

    # dashboard_job requires the location of the evaluation database.
    "dashboard_job.yaml": ["+eval_db_uri=???"],

    # replay_job requires the location of the replay database.
    "replay_job.yaml": ["+db_uri=???"],
}

@pytest.mark.parametrize("config_name", get_entrypoint_configs())
def test_entrypoint_config_composition(config_name: str):
    """
    Tests that a given entrypoint configuration can be composed successfully and
    that all `_target_` fields resolve to valid, importable classes.
    This simulates a real hydra run for the main job configs by providing
    the minimum required runtime overrides.
    """
    try:
        with hydra.initialize(version_base=None, config_path="../configs"):
            overrides = REQUIRED_RUNTIME_OVERRIDES.get(config_name, [])
            cfg = hydra.compose(config_name=config_name, overrides=overrides)

            # To keep the test fast and focused on config correctness, we won't fully
            # instantiate the object graph. Instead, we'll recursively traverse the
            # composed config and verify that every '_target_' string points to a
            # class that can be imported. This catches typing and path errors early.
            def check_targets(node):
                if isinstance(node, dict) and "_target_" in node:
                    hydra.utils.get_class(node["_target_"])
                elif isinstance(node, dict):
                    for v in node.values():
                        check_targets(v)
                elif isinstance(node, list):
                    for v in node:
                        check_targets(v)

            # We use resolve=False because resolving interpolations might require
            # other runtime values we don't have. We only care about the structure
            # and the validity of _target_ paths.
            check_targets(OmegaConf.to_container(cfg, resolve=False))

    except Exception as e:
        pytest.fail(
            f"Failed to compose or validate _target_s in '{config_name}': {e}"
        )
