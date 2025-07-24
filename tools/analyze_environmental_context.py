#!/usr/bin/env python3
"""
Script to analyze learned environmental context embeddings.

This script loads a trained model with environmental context and analyzes:
1. The learned task embeddings
2. Similarities between different task embeddings
3. Visualization of the embedding space
"""

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from metta.agent.policy_store import PolicyStore


def load_policy_with_context(policy_path: str) -> torch.nn.Module:
    """Load a policy that has environmental context embeddings."""
    # Load the policy
    policy_store = PolicyStore(OmegaConf.create({"run_dir": os.path.dirname(policy_path)}))
    policy_record = policy_store.policy_record(policy_path)

    if not policy_record:
        raise ValueError(f"Could not load policy from {policy_path}")

    policy = policy_record.policy

    # Check if the policy has environmental context
    if not hasattr(policy, "components") or "environmental_context_embedding" not in policy.components:
        raise ValueError("Policy does not have environmental context embeddings")

    return policy


def extract_task_embeddings(policy: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Extract all task embeddings from the policy."""
    context_layer = policy.components["environmental_context_embedding"]
    embeddings = context_layer.get_all_embeddings().cpu().numpy()

    # Create a mapping of task names to embeddings
    # For now, we'll use the embedding index as the task ID
    # In a real scenario, you'd want to map these to actual task names
    task_embeddings = {}
    for i in range(embeddings.shape[0]):
        task_embeddings[f"task_{i}"] = embeddings[i]

    return task_embeddings


def compute_embedding_similarities(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute cosine similarities between all task embeddings."""
    embedding_matrix = np.array(list(embeddings.values()))
    similarities = cosine_similarity(embedding_matrix)
    return similarities


def visualize_embedding_similarities(similarities: np.ndarray, task_names: List[str], output_path: str):
    """Create a heatmap of embedding similarities."""
    plt.figure(figsize=(12, 10))
    plt.imshow(similarities, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.title("Task Embedding Similarities")
    plt.xlabel("Task Index")
    plt.ylabel("Task Index")

    # Add task labels if we have them
    if len(task_names) <= 20:  # Only show labels if not too many tasks
        plt.xticks(range(len(task_names)), task_names, rotation=45, ha="right")
        plt.yticks(range(len(task_names)), task_names)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_embedding_space(embeddings: Dict[str, np.ndarray], output_path: str, method: str = "pca"):
    """Visualize the embedding space using PCA or t-SNE."""
    embedding_matrix = np.array(list(embeddings.values()))
    task_names = list(embeddings.keys())

    if method == "pca":
        reducer = PCA(n_components=2)
        title = "Task Embeddings (PCA)"
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        title = "Task Embeddings (t-SNE)"
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    reduced_embeddings = reducer.fit_transform(embedding_matrix)

    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

    # Add task labels
    for i, task_name in enumerate(task_names):
        plt.annotate(
            task_name,
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_embedding_statistics(embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute statistics about the embeddings."""
    embedding_matrix = np.array(list(embeddings.values()))

    stats = {
        "num_tasks": len(embeddings),
        "embedding_dim": embedding_matrix.shape[1],
        "mean_norm": np.mean(np.linalg.norm(embedding_matrix, axis=1)),
        "std_norm": np.std(np.linalg.norm(embedding_matrix, axis=1)),
        "mean_similarity": np.mean(cosine_similarity(embedding_matrix)),
        "std_similarity": np.std(cosine_similarity(embedding_matrix)),
    }

    return stats


def save_analysis_results(results: Dict, output_path: str):
    """Save analysis results to a file."""
    with open(output_path, "w") as f:
        f.write("Environmental Context Embedding Analysis\n")
        f.write("=" * 50 + "\n\n")

        f.write("Embedding Statistics:\n")
        for key, value in results["statistics"].items():
            f.write(f"  {key}: {value}\n")

        f.write("\nTop Similar Task Pairs:\n")
        similarities = results["similarities"]
        task_names = list(results["embeddings"].keys())

        # Find top similar pairs
        pairs = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                pairs.append((similarities[i, j], task_names[i], task_names[j]))

        pairs.sort(reverse=True)
        for i, (sim, task1, task2) in enumerate(pairs[:10]):
            f.write(f"  {i + 1}. {task1} <-> {task2}: {sim:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze environmental context embeddings")
    parser.add_argument("policy_path", help="Path to the trained policy")
    parser.add_argument("--output_dir", default="./embedding_analysis", help="Directory to save analysis results")
    parser.add_argument(
        "--visualization_method",
        choices=["pca", "tsne"],
        default="pca",
        help="Method for embedding space visualization",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading policy from {args.policy_path}")
    policy = load_policy_with_context(args.policy_path)

    print("Extracting task embeddings...")
    embeddings = extract_task_embeddings(policy)

    print("Computing embedding similarities...")
    similarities = compute_embedding_similarities(embeddings)

    print("Computing embedding statistics...")
    statistics = analyze_embedding_statistics(embeddings)

    # Create visualizations
    print("Creating visualizations...")
    task_names = list(embeddings.keys())

    # Similarity heatmap
    similarity_path = os.path.join(args.output_dir, "embedding_similarities.png")
    visualize_embedding_similarities(similarities, task_names, similarity_path)

    # Embedding space visualization
    embedding_path = os.path.join(args.output_dir, f"embedding_space_{args.visualization_method}.png")
    visualize_embedding_space(embeddings, embedding_path, args.visualization_method)

    # Save analysis results
    results = {"embeddings": embeddings, "similarities": similarities, "statistics": statistics}

    results_path = os.path.join(args.output_dir, "analysis_results.txt")
    save_analysis_results(results, results_path)

    # Print summary
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Number of tasks: {statistics['num_tasks']}")
    print(f"Embedding dimension: {statistics['embedding_dim']}")
    print(f"Mean embedding norm: {statistics['mean_norm']:.4f}")
    print(f"Mean similarity: {statistics['mean_similarity']:.4f}")

    # Show top similar pairs
    print("\nTop similar task pairs:")
    pairs = []
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            pairs.append((similarities[i, j], task_names[i], task_names[j]))

    pairs.sort(reverse=True)
    for i, (sim, task1, task2) in enumerate(pairs[:5]):
        print(f"  {i + 1}. {task1} <-> {task2}: {sim:.4f}")


if __name__ == "__main__":
    main()
