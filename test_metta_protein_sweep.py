import yaml

from metta.rl.carbs.metta_protein import MettaProtein


def coerce_numbers(d):
    if isinstance(d, dict):
        return {k: coerce_numbers(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [coerce_numbers(x) for x in d]
    elif isinstance(d, str):
        # Try to convert to int or float
        try:
            if "." in d or "e" in d or "E" in d:
                return float(d)
            else:
                return int(d)
        except ValueError:
            return d
    else:
        return d


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)["sweep"]
        cfg = coerce_numbers(cfg)
        # Convert to DictConfig for type compatibility
        from omegaconf import OmegaConf

        return OmegaConf.create(cfg)


def main():
    cfg = load_cfg("configs/sweep/minimal_protein_sweep.yaml")
    # Test with wandb_run parameter (stubbed)
    protein = MettaProtein(cfg, wandb_run=None)
    print("✅ Constructor works with wandb_run parameter")

    params, info = protein.suggest()
    print("✅ suggest() method works")
    print("Suggested params:", params)

    # Test observe method
    protein.observe(params, score=1.0, cost=0.5)
    print("✅ observe() method works")

    # Test static method
    MettaProtein._record_observation(None, 1.0, 0.5)
    print("✅ _record_observation() static method works")

    print("✅ All stubbed WandB methods working!")


if __name__ == "__main__":
    main()
