import time

import torch
from tensordict import TensorDict


# Simulate the pipeline with nested tensors
def create_test_data(batch_size=32, max_seq_len=300, device="cuda"):
    """Create test data simulating ragged sequences with padding."""
    observations = []
    actual_lengths = []

    for i in range(batch_size):
        # Random actual sequence length (simulate varying lengths)
        actual_len = torch.randint(50, 250, (1,)).item()
        actual_lengths.append(actual_len)

        # Create observation: [seq_len, 3] where 3 = (coord, attr_idx, attr_val)
        obs = torch.zeros(max_seq_len, 3, device=device)

        # Fill actual data
        obs[:actual_len, 0] = torch.randint(0, 255, (actual_len,), device=device)  # coords
        obs[:actual_len, 1] = torch.randint(0, 10, (actual_len,), device=device)  # attr indices
        obs[:actual_len, 2] = torch.randn(actual_len, device=device) * 10  # attr values

        # Fill padding with 255 for coords
        obs[actual_len:, 0] = 255

        observations.append(obs)

    # Stack into batch tensor
    observations = torch.stack(observations)
    return observations, actual_lengths


def benchmark_nested_vs_padded():
    """Compare performance of nested tensor vs padded tensor processing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Create test data
    batch_size = 128
    observations, actual_lengths = create_test_data(batch_size, device=device)

    # Create TensorDict
    td = TensorDict(
        {
            "x": observations,
        },
        batch_size=batch_size,
    )

    # Test with regular padded approach
    print("\n=== Testing Padded Tensor Approach ===")
    from metta.agent.lib.obs_tokenizers import ObsTokenPadStrip

    padded_layer = ObsTokenPadStrip(obs_shape=(300, 3))
    padded_layer.to(device)

    # Warm up
    _ = padded_layer(td.clone())

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()

    padded_output = padded_layer(td.clone())

    torch.cuda.synchronize() if device == "cuda" else None
    padded_time = time.time() - start_time

    print(f"Padded approach time: {padded_time:.4f}s")
    print(f"Output shape: {padded_output[padded_layer._name].shape}")
    print(f"Mask shape: {padded_output['obs_mask'].shape}")

    # Test with nested tensor approach
    print("\n=== Testing Nested Tensor Approach ===")
    from metta.agent.lib.obs_tokenizers import ObsTokenPadStripNested

    nested_layer = ObsTokenPadStripNested(obs_shape=(300, 3))
    nested_layer.to(device)

    # Warm up
    _ = nested_layer(td.clone())

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()

    nested_output = nested_layer(td.clone())

    torch.cuda.synchronize() if device == "cuda" else None
    nested_time = time.time() - start_time

    print(f"Nested approach time: {nested_time:.4f}s")
    print(f"Output type: {type(nested_output[nested_layer._name])}")
    print(f"Is nested: {nested_output.get('is_nested', False)}")

    # Show sequence length statistics
    seq_lengths = nested_output["seq_lengths"]
    print("\nSequence length statistics:")
    print(f"  Min length: {seq_lengths.min().item()}")
    print(f"  Max length: {seq_lengths.max().item()}")
    print(f"  Mean length: {seq_lengths.float().mean().item():.1f}")
    print(f"  Total tokens (nested): {seq_lengths.sum().item()}")
    print(f"  Total tokens (padded): {batch_size * padded_output[padded_layer._name].shape[1]}")
    print(
        f"  Wasted computation: {(1 - seq_lengths.sum().item() / (batch_size * padded_output[padded_layer._name].shape[1])) * 100:.1f}%"
    )


def test_full_pipeline():
    """Test the full pipeline with nested tensors."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== Testing Full Pipeline with Nested Tensors ===")
    print(f"Device: {device}")

    # Import all layers
    from metta.agent.lib.obs_enc import ObsLatentAttn, ObsVanillaAttn
    from metta.agent.lib.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStripNested

    # Create test data
    batch_size = 32
    observations, _ = create_test_data(batch_size, device=device)

    td = TensorDict(
        {
            "x": observations,
        },
        batch_size=batch_size,
    )

    # Build pipeline
    tokenizer = ObsTokenPadStripNested(obs_shape=(300, 3))
    normalizer = ObsAttrValNorm(feature_normalizations=[1.0] * 10)
    embedder = ObsAttrEmbedFourier(atr_embed_dim=12, num_freqs=8)
    latent_attn = ObsLatentAttn(
        out_dim=32,
        use_mask=False,  # Not needed with nested tensors
        num_query_tokens=7,
        query_token_dim=32,
        num_heads=8,
        num_layers=3,
    )
    vanilla_attn = ObsVanillaAttn(out_dim=128, num_heads=8, num_layers=3, use_mask=False, use_cls_token=True)

    # Move to device
    for layer in [tokenizer, normalizer, embedder, latent_attn, vanilla_attn]:
        layer.to(device)

    # Forward pass
    print("\nRunning forward pass...")
    td = tokenizer(td)
    print(f"After tokenizer - is_nested: {td.get('is_nested', False)}")

    # Set source names for subsequent layers
    normalizer._sources = [{"name": tokenizer._name}]
    td = normalizer(td)
    print(f"After normalizer - is_nested: {td.get('is_nested', False)}")

    embedder._sources = [{"name": normalizer._name}]
    td = embedder(td)
    print(f"After embedder - is_nested: {td.get('is_nested', False)}")

    latent_attn._sources = [{"name": embedder._name}]
    td = latent_attn(td)
    print(f"After latent_attn - is_nested: {td.get('is_nested', False)}")
    print(f"Latent attention output shape: {td[latent_attn._name].shape}")

    vanilla_attn._sources = [{"name": latent_attn._name}]
    td = vanilla_attn(td)
    print(f"After vanilla_attn - is_nested: {td.get('is_nested', False)}")
    print(f"Final output shape: {td[vanilla_attn._name].shape}")

    print("\nPipeline executed successfully!")
    print(f"Final output: [batch_size={batch_size}, out_dim={td[vanilla_attn._name].shape[-1]}]")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_nested_vs_padded()

    # Test full pipeline
    test_full_pipeline()
