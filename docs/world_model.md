
# World Model Implementation

## Idea

The core idea behind the world model is to create a compressed representation of the environment's observations. Instead of feeding high-dimensional raw observations directly to the agent, we first train a world model to encode these observations into a lower-dimensional latent space. The agent then uses this compressed latent representation for training and decision-making.

The goal is to:

1.  **Improve Sample Efficiency:** By learning a compressed representation of the world, the agent can potentially learn faster and generalize better.
2.  **Enable Imagination and Planning:** A world model can be used to simulate future states, allowing the agent to "imagine" and plan its actions without interacting with the real environment. (This is a future goal, not yet implemented).
3.  **Reduce Computational Load:** The agent's policy network processes smaller, denser latent vectors instead of large raw observations, which can reduce the computational cost of training and inference.

## Implementation

The implementation of the world model can be broken down into the following key components:

### 1. World Model Architecture

The world model is an autoencoder with the following architecture:

*   **Encoder:**
    *   Input: Raw observation (200x3)
    *   Layers:
        *   Linear(200\*3 -> 2048)
        *   Linear(2048 -> 2048)
        *   Linear(2048 -> 1024)
        *   Linear(1024 -> 128) - This is the latent space.
*   **Decoder:**
    *   Input: Latent vector (128)
    *   Layers:
        *   Linear(128 -> 1024)
        *   Linear(1024 -> 2048)
        *   Linear(2048 -> 2048)
        *   Linear(2048 -> 200\*3) - Reconstructed observation.

The latent space dimension was initially 16 and was later increased to 128 to better match the LSTM input size in the agent's policy.

### 2. Pre-training

The world model is pre-trained using supervised learning before the agent starts training. The pre-training process is as follows:

1.  **Data Collection:** The trainer collects raw observations from the environment.
2.  **Reconstruction:** The world model takes a raw observation, encodes it into a latent vector, and then decodes it back to a reconstructed observation.
3.  **Loss Calculation:** The mean squared error (MSE) between the original and reconstructed observation is calculated.
4.  **Optimization:** The world model's parameters are updated using the Adam optimizer to minimize the reconstruction loss.

This pre-training is configured to run for 100 million steps to ensure the world model learns a good representation of the environment.

### 3. Integration with the Agent

Once the world model is pre-trained, it is integrated with the agent's training process:

1.  **Observation Encoding:** During the main training loop, raw observations from the environment are first passed through the world model's encoder to get the latent representation.
2.  **Agent Input:** This latent vector is then used as the input to the agent's policy network.
3.  **Policy Network:** The agent's policy network (`Fast` component policy) was modified to accept the latent observations directly, bypassing the CNN pipeline that was used for raw observations.
4.  **Experience Spec:** The agent's experience spec was updated to use `latent_obs` instead of `env_obs`.

### 4. Monitoring and Logging

To monitor the pre-training process, the following were added:

*   **Wandb Logging:** The reconstruction loss and pre-training steps are logged to Weights & Biases every 100 steps.
*   **Console Logging:** The pre-training progress is logged to the console every 1000 steps.
*   **Heartbeat Monitoring:** A heartbeat is recorded during the pre-training loop to prevent timeouts during the long pre-training phase.

## Goal

The ultimate goal of the world model is to improve the agent's performance and sample efficiency. By learning a compressed and informative representation of the world, the agent can focus on learning the policy without being burdened by the high dimensionality of the raw observations.

The current implementation focuses on the first step: learning the world model and using its latent representation for the agent. Future work could involve using the world model for imagination and planning, which could lead to even greater performance improvements.
