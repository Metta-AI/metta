import wandb
from wandb.sdk.wandb_run import Run

run_name = "metta-research/metta/dd.navsequencemem.smallinventory"

api = wandb.Api()
run: Run = api.run(run_name)

# history_iter = run.scan_history()
# history = [row for row in history_iter]
# print(history)
