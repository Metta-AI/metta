import torch
from tensordict import TensorDict

# Assuming RobustObsEncoderRev1 is importable from its location
# Adjust the import path as necessary
from metta.agent.lib.obs_encoders_old import RobustObsEncoderRev1

# --- Configuration ---
cfg = {
    "sources": [
        {"name": "obs", "mode": "input"},
        {"name": "center_pixels", "mode": "input"},
    ],
    "name": "encoded_observation",
}

# --- Instantiate the Encoder ---
# Ensure necessary parameters like dimensions are accessible or hardcoded
# Based on RobustObsEncoderRev1:
center_pixel_len = 34
lstm_h_len = 128
lstm_h_layers = 2
grid_h, grid_w = 11, 11
num_channels = 34  # Derived from max channel index + 1 in channel_sets

encoder = RobustObsEncoderRev1(**cfg)

# --- Create Sample Data ---
N = 2  # Batch size

# Dummy observation tensor
# Shape: [N, C, H, W]
obs_tensor = torch.rand(N, num_channels, grid_h, grid_w) * 10  # Some non-zero values

# Dummy center pixels tensor
# Shape: [N, center_pixel_len]
center_pixels_tensor = torch.rand(N, center_pixel_len)

# Dummy previous LSTM hidden state
# Shape: [lstm_h_layers, N, lstm_h_len] - Note: LayerBase expects [layers, N, dim]
# The encoder specifically uses state_h_prev[self.lstm_h_layers - 1]
# Shape becomes [N, lstm_h_len] after selection inside _forward
state_h_prev_tensor = torch.rand(lstm_h_layers, N, lstm_h_len)

# --- Package into TensorDict ---
td = TensorDict(
    {
        "obs": obs_tensor,
        "center_pixels": center_pixels_tensor,
        # The encoder looks for "state_h_prev" directly in the td
        "state_h_prev": state_h_prev_tensor,
    },
    batch_size=[N],
)

# --- Device Selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Move Model and Data to Device ---
encoder = encoder.to(device)
td = td.to(device)

# --- Setup the Encoder ---
# This initializes internal states required by the forward pass
print("Setting up encoder...")
encoder.setup()  # Call setup before forward
print("Encoder setup complete.")

# --- Run Forward Pass ---
print("Running forward pass...")
try:
    # The LayerBase __call__ method handles the forward pass
    output_td = encoder(td)
    output_tensor = output_td[encoder._name]  # Access output using the configured name
    print("Forward pass successful!")
    print(f"Output tensor shape: {output_tensor.shape}")
    # Expected output shape: [N, hidden] where hidden = key_dim * 8 = 16 * 8 = 128
    expected_shape = (N, encoder.hidden)
    print(f"Expected shape: {expected_shape}")
    assert output_tensor.shape == expected_shape, "Output shape mismatch!"

except Exception as e:
    print(f"An error occurred during the forward pass: {e}")
    import traceback

    traceback.print_exc()

print("Script finished.")
