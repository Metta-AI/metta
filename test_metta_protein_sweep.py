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
        return coerce_numbers(cfg)


def main():
    cfg = load_cfg("configs/sweep/minimal_protein_sweep.yaml")
    protein = MettaProtein(cfg)
    params, info = protein.suggest(fill=None)
    print("Suggested params:", params)
    # Optionally, observe a fake result
    protein.observe(params, score=1.0, cost=0.5)


if __name__ == "__main__":
    main()
