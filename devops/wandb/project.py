import argparse
import yaml
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_user', required=True)
    parser.add_argument('--wandb_project', required=True)
    parser.add_argument('--yaml_path', required=True)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    api = wandb.Api()
    run = api.runs(f"{args.wandb_user}/{args.wandb_project}")

    runs = api.runs(f"{args.wandb_user}/{args.wandb_project}")
    for run in runs:
        if args.load:
            with open(args.yaml_path, 'r') as f:
                view = yaml.safe_load(f)
                run.config.update(view)

        if args.save:
            view = dict(run.config)
            with open(args.yaml_path, 'w') as f:
                yaml.safe_dump(view, f)

if __name__ == "__main__":
    main()
