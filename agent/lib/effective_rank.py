import torch
import wandb
import sys
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # pickle needs the 'agent' module to exist on the python path
sys.path.append(root_dir) 

ENTITY_NAME = "metta-research"
PROJECT_NAME = "metta"

def batch_srank(artifact_project: str, version_start: int, version_end: int, skip_every: int, delta: float):
    '''
    Downloads the model from wandb, computes the effective rank, and saves the results: a model.pt, a plot, and a csv.
    '''
    run = wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)
    ranks = []
    versions = []
    for version in range(version_start, version_end, skip_every):
        artifact_path = f"/Users/alex/Software/Stem/metta/artifacts/{artifact_project}:v{version}/model.pt"
        if not os.path.exists(artifact_path):
            dl_model(run, artifact_project, version)

        model = torch.load(artifact_path, weights_only=False, map_location=torch.device('cpu')) 
        # TODO: make the below not-hardcoded. We want the penultimate layer.
        # linear_layer = model._policy.policy.policy._agent._encoder.network[5].weight # ._critic_linear.weight critic or the critic
        # Example: Iterating over all modules in a model
        # print(model._policy.policy.policy._agent._critic_linear.weight.data)
        print(f"Shape of linear_layer weights: {model._policy.policy.policy._agent._critic_linear.weight.data.shape}")
        # print(f"Shape of linear_layer weights: {model._policy.policy.policy._agent._encoder.weight.data.shape}")
        print(f"Shape of linear_layer weights: {model._policy.policy.policy._agent._decoder.shape}")
        linear_layer = model._policy.policy.policy._agent._critic_linear.weight.data
        rank = compute_effective_rank(linear_layer, delta)
        ranks.append(rank)
        versions.append(version)
    max_possible_rank = min(linear_layer.shape)
    save_srank_plot(ranks, versions, max_possible_rank, delta, artifact_project)
    save_srank_csv(ranks, versions, delta, artifact_project)

def compute_effective_rank(matrix: torch.Tensor, delta: float = 0.01):
    """
    Computes the effective rank of a matrix based on the given delta value.
    Effective rank formula:
    srank_\delta(\Phi) = min{k: sum_{i=1}^k σ_i / sum_{j=1}^d σ_j ≥ 1 - δ}
    See the paper titled 'Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning' by A. Kumar et al.
    """
    # Singular value decomposition. We only need the singular value matrix.
    _, S, _ = torch.linalg.svd(matrix)
    
    # Calculate the cumulative sum of singular values
    total_sum = S.sum()
    cumulative_sum = torch.cumsum(S, dim=0)
    
    # Find the smallest k that satisfies the effective rank condition
    threshold = (1 - delta) * total_sum
    effective_rank = torch.where(cumulative_sum >= threshold)[0][0].item() + 1  # Add 1 for 1-based indexing
    
    return effective_rank

def dl_model(run, artifact_project: str, version: int):
    artifact_name = f'{ENTITY_NAME}/{PROJECT_NAME}/{artifact_project}:v{version}'
    try:
        print(f"Attempting to access artifact {artifact_name}")
        artifact = run.use_artifact(artifact_name)
    except wandb.errors.CommError as e:
        print(f"Error accessing artifact {artifact_name}: {e}")
        raise e
    print(f"Successfully accessed artifact {artifact_name}!")
    # artifact = run.use_artifact(artifact_name)


def save_srank_plot(ranks: list[int], versions: list[int], max_possible_rank: int, delta: float, artifact_project: str):
    plt.figure(figsize=(10, 6))
    plt.plot(versions, ranks, marker='o', linestyle='-')
    plt.title(f'Effective Rank vs. Version (delta={delta})\n{artifact_project}', fontsize=14)
    plt.xlabel('Version', fontsize=12)
    plt.ylabel('Effective Rank', fontsize=12)
    plt.ylim(0, max_possible_rank + 5)
    plt.grid(True)
    plt.savefig(f'effective_rank_vs_version_delta={delta}.png')

def save_srank_csv(ranks: list[int], versions: list[int], delta: float, artifact_project: str):
    df = pd.DataFrame({'Version': versions, 'Effective Rank': ranks})
    df.to_csv(f'effective_rank_data_delta_{delta}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact_project', required=True, help='The project containing the artifacts e.g. p2.train.norm.feat')
    parser.add_argument('--version_start', type=int, required=True, help='The starting version number of the artifacts to evaluate.')
    parser.add_argument('--version_end', type=int, required=True, help='The ending version number of the artifacts to evaluate.')
    parser.add_argument('--skip_every', type=int, required=True, help='The number of versions to skip between each version.')

    args = parser.parse_args()

    deltas = [0.05, 0.02, 0.01]

    for delta in deltas:
        batch_srank(args.artifact_project, args.version_start, args.version_end, args.skip_every, delta)


# def dl_and_calc_effective_rank(artifact_project: str, version_start: int, version_end: int, skip_every: int, delta: float):
#     run = wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)

#     ranks = []
#     versions = []

#     for version in range(version_start, version_end, skip_every):
#         artifact_name = f'{ENTITY_NAME}/{PROJECT_NAME}/{artifact_project}:v{version}'
#         artifact = run.use_artifact(artifact_name)

#         artifact_path = f"/Users/alex/Software/Stem/metta/artifacts/{artifact_project}:v{version}/model.pt"
#         if not os.path.exists(artifact_path):
#             artifact.download() #later, try to use path_prefix or similar to only pull the matrix needed
#         model = torch.load(artifact_path, weights_only=False, map_location=torch.device('cpu')) 

#         linear_layer = model._policy.policy.policy._agent._encoder.network[5].weight
#         print(linear_layer)

#         if linear_layer.dim() == 2:
#             rank = compute_effective_rank(linear_layer, delta)
#             print(f"Effective Rank of linear_layer: {rank}")
#             ranks.append(rank)
#             versions.append(version)
#         else:
#             print("The linear_layer is not a 2D tensor and cannot be processed by effective_rank.")

#     max_possible_rank = min(linear_layer.shape)
#     plt.figure(figsize=(10, 6))
#     plt.plot(versions, ranks, marker='o', linestyle='-')
#     plt.title(f'Effective Rank vs. Version (delta={delta})\n{artifact_project}', fontsize=14)
#     plt.xlabel('Version', fontsize=12)
#     plt.ylabel('Effective Rank', fontsize=12)
#     plt.ylim(0, max_possible_rank + 5)
#     plt.grid(True)
#     plt.savefig(f'effective_rank_vs_version_delta={delta}.png')
    
#     import pandas as pd
#     df = pd.DataFrame({'Version': versions, 'Effective Rank': ranks})
#     df.to_csv(f'effective_rank_data_delta_{delta}.csv', index=False)
#     plt.show()
