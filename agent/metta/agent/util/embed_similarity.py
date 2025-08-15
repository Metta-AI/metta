import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# # --- The code to drop into the action.py file's forward pass ---

# # Specify the filename where you want to save the embeddings
# embedding_output = self._net(self.active_indices)

# output_filename = 'agent/util/embedding_output.npy'

# print(f"Preparing to save embeddings to {output_filename}...")
# print(f"Original tensor device: {embedding_output.device}")
# print(f"Original tensor requires_grad: {embedding_output.requires_grad}")

# # 1. Detach the tensor from the computation graph (if it has gradients)
# # 2. Move the tensor to the CPU (NumPy can't directly handle GPU tensors)
# # 3. Convert the tensor to a NumPy array
# try:
#     numpy_embeddings = embedding_output.detach().cpu().numpy()

#     # 4. Save the NumPy array to a .npy file
#     np.save(output_filename, numpy_embeddings)

#     print(f"Successfully saved embeddings with shape {numpy_embeddings.shape} to {output_filename}")

# except Exception as e:
#     print(f"An error occurred while saving the embeddings: {e}")

# # --- End of drop-in code ---


# --- Step 1: Load your embeddings ---
# Assuming 'your_embeddings_file.npy' exists from the previous step
try:
    embeddings_array = np.load("agent/util/embedding_output.npy")
    print(f"Loaded embeddings with shape: {embeddings_array.shape}")
    # Make sure it matches the expected (25, 16)
    if embeddings_array.shape != (25, 16):
        print(f"Warning: Loaded shape {embeddings_array.shape} doesn't match expected (25, 16)")
except FileNotFoundError:
    print("Error: Embedding file not found. Please ensure 'your_embeddings_file.npy' exists.")
    # Create dummy data for demonstration if file not found
    print("Using dummy data (25, 16) for demonstration.")
    embeddings_array = np.random.rand(25, 16).astype(np.float32)


# Make sure we have something to work with
if "embeddings_array" in locals() and embeddings_array.size > 0:
    num_embeddings, embedding_dimension = embeddings_array.shape

    # --- Calculate Pairwise Similarity/Distance Matrices ---
    # (This part is the same as before)
    cosine_sim_matrix = cosine_similarity(embeddings_array)
    euclidean_dist_matrix = euclidean_distances(embeddings_array)

    # --- NEW: Technique 1 - Heatmap Visualization ---
    print("\nGenerating heatmaps...")

    # Plotting Cosine Similarity Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cosine_sim_matrix, cmap="viridis", annot=False
    )  # annot=True if you want numbers (maybe too crowded for 25x25)
    plt.title("Cosine Similarity Between All Embeddings")
    plt.xlabel("Embedding Index")
    plt.ylabel("Embedding Index")
    plt.show()

    # Plotting Euclidean Distance Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(euclidean_dist_matrix, cmap="magma_r", annot=False)  # '_r' reverses the map (lower distance = brighter)
    plt.title("Euclidean Distance Between All Embeddings")
    plt.xlabel("Embedding Index")
    plt.ylabel("Embedding Index")
    plt.show()

    # --- NEW: Technique 2 - Sorted Neighbor Lists ---
    print("\nGenerating sorted neighbor lists (Top 5)...")

    n_neighbors = 5  # How many neighbors to show (excluding self)

    print("\n--- Top Neighbors by Cosine Similarity (Higher is Better) ---")
    # Argsort gives the indices that would sort the array.
    # We negate the similarity matrix because argsort sorts ascendingly, and we want descending similarity.
    cosine_neighbor_indices = np.argsort(-cosine_sim_matrix, axis=1)

    for i in range(num_embeddings):
        neighbors = []
        for j in range(1, n_neighbors + 1):  # Start from 1 to skip self (index 0 is always the item itself)
            neighbor_idx = cosine_neighbor_indices[i, j]
            similarity = cosine_sim_matrix[i, neighbor_idx]
            neighbors.append(f"Idx {neighbor_idx} ({similarity:.3f})")
        print(f"Index {i}: {', '.join(neighbors)}")

    print("\n--- Top Neighbors by Euclidean Distance (Lower is Better) ---")
    # Argsort sorts ascendingly, which is perfect for distance (lower is better)
    euclidean_neighbor_indices = np.argsort(euclidean_dist_matrix, axis=1)

    for i in range(num_embeddings):
        neighbors = []
        for j in range(1, n_neighbors + 1):  # Start from 1 to skip self
            neighbor_idx = euclidean_neighbor_indices[i, j]
            distance = euclidean_dist_matrix[i, neighbor_idx]
            neighbors.append(f"Idx {neighbor_idx} ({distance:.3f})")
        print(f"Index {i}: {', '.join(neighbors)}")

else:
    print("Could not load or generate embeddings. Cannot proceed with analysis.")
