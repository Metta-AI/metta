import os

import cv2
import numpy as np
from huggingface_hub import HfApi, Repository, repocard, upload_folder

from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log, project_tmp_dir
import imageio

# https://huggingface.co/metta-ai/baseline.sf.v0.6.1/resolve/main/behaviors_resources_100x100.gif

def generate_model_card(
    dir_path: str,
    algo: str,
    env: str,
    repo_id: str,
    rewards: list = None,
    enjoy_name: str = None,
    train_name: str = None,
):
    readme_path = os.path.join(dir_path, "README.md")
    repo_name = repo_id.split("/")[1]

    readme = f"""
A(n) **{algo}** model trained on the **{env}** environment.\n
This model was trained using Sample-Factory 2.0: https://github.com/alex-petrenko/sample-factory.
Documentation for how to use Sample-Factory can be found at https://www.samplefactory.dev/\n\n
## Downloading the model\n
After installing Sample-Factory, download the model with:
```
python -m sample_factory.huggingface.load_from_hub -r {repo_id}
```\n
    """

    if enjoy_name is None:
        enjoy_name = "<path.to.enjoy.module>"

    readme += f"""
## Using the model\n
To run the model after download, use the `enjoy` script corresponding to this environment:
```
python -m {enjoy_name} --algo={algo} --env={env} --train_dir=./train_dir --experiment={repo_name}
```
\n
You can also upload models to the Hugging Face Hub using the same script with the `--push_to_hub` flag.
See https://www.samplefactory.dev/10-huggingface/huggingface/ for more details
    """

    if train_name is None:
        train_name = "<path.to.train.module>"

    readme += f"""
## Training with this model\n
To continue training with this model, use the `train` script corresponding to this environment:
```
python -m {train_name} --algo={algo} --env={env} --train_dir=./train_dir --experiment={repo_name} --restart_behavior=resume --train_for_env_steps=10000000000
```\n
Note, you may have to adjust `--train_for_env_steps` to a suitably high number as the experiment will resume at the number of steps it concluded at.
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {}
    metadata["library_name"] = "sample-factory"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "sample-factory",
    ]

    if rewards is not None:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        eval = repocard.metadata_eval_result(
            model_pretty_name=algo,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env,
            dataset_id=env,
        )

        metadata = {**metadata, **eval}

    repocard.metadata_save(readme_path, metadata)


def push_to_hf(dir_path: str, repo_name: str):
    repo_url = HfApi().create_repo(
        repo_id=repo_name,
        private=False,
        exist_ok=True,
    )

    upload_folder(
        repo_id=repo_name,
        folder_path=dir_path,
        path_in_repo=".",
        ignore_patterns=[".git/*"],
    )

    log.info(f"The model has been pushed to {repo_url}")


def load_from_hf(dir_path: str, repo_id: str):
    temp = repo_id.split("/")
    repo_name = temp[1]

    local_dir = os.path.join(dir_path, repo_name)
    Repository(local_dir, repo_id)
    log.info(f"The repository {repo_id} has been cloned to {local_dir}")
