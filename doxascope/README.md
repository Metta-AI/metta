# Doxascope

The "doxascope" is a tool designed to investigate the internal states of a reinforcement learning agent. It consists of a neural network that takes an agent's LSTM memory vectors as input and attempts to predict the agent's future actions.

This directory contains the necessary scripts for data logging, preprocessing, training, and analysis of the doxascope network outputs.


## Files

-   `doxascope_data.py`: Contains the `DoxascopeLogger` for capturing agent memory vectors and positions during a simulation, and the `preprocess_doxascope_data` function for preparing the raw data for training.
-   `doxascope_train.py`: A command-line script to initiate the training process.
-   `doxascope_network.py`: Defines the `DoxascopeNet` architecture and the `DoxascopeTrainer` class that handles the training, validation, and analysis loop.
-   `doxascope_analysis.py`: Includes functions for generating plots, such as training curves, confusion matrices, and multi-step accuracy graphs.
-   `doxascope_sweep.py`: A script for running hyperparameter sweeps to find optimal model configurations.

## Network Architecture

The `DoxascopeNet` has a modular architecture that can be changed as the user wishes. The default architecture, based on sweeps over architecture configurations is:

-   **Input Layer**: Takes the full concatenated memory vector as input.

-   **Parallel State Processors**: The input vector is immediately split back into its constituent `h` and `c` states. Each state is passed through its own small, independent multi-layer perceptron (MLP). This initial parallel processing was found to improve performance, as it allows the network to learn features specific to each part of the LSTM memory before combining them.

-   **Main Processing Network**: The outputs from the two state processors are concatenated and fed into a larger, deeper 3 layer main MLP. This network performs the core feature extraction on the combined memory representation.

-   **Multi-Timestep Prediction Heads**: The final layer consists of multiple independent prediction heads. There is one head for each past and future timestep the network is configured to predict (e.g., for `t-2, t-1, t+1, t+2, t+3`). Each head is a linear layer that outputs the final classification logits for the five movement classes.

-   **Residual Skip Connection**: A skip connection takes the original, raw input vector and adds it directly to the output of the first *future* prediction head (`t+1`). This residual link helps with gradient flow and training stability.

## Usage

### 1. Data Collection

To enable data collection, you need to add the `doxascope` configuration to your user config file (e.g., `metta/configs/user/your_name.yaml`). Add the following snippet to enable the logger:

```yaml
sim_job:
  simulation_suite:
    doxascope:
      enabled: true
```

When you run a simulation (i.e., an evaluation) with this configuration, the `DoxascopeLogger` will automatically be activated. It will record the LSTM memory vectors and positions for each agent at every timestep.

The raw data is saved in `doxascope/data/raw_data/<policy_name>/`. The `<policy_name>` is derived from the policy URI specified in your config.

For each simulation, a new JSON file named `doxascope_data_{simulation_id}.json` is created, where `simulation_id` is a unique identifier for that run.

### 2. Training the Doxascope Network

Once you have collected data, you can train the network using the `doxascope_train.py` script.

**Usage:**

```bash
python -m doxascope.doxascope_train <policy_name> <num_future_timesteps> [options]
```

**Arguments:**

-   `<policy_name>`: The name of the policy whose data you want to use for training. This corresponds to the subdirectory within `doxascope/data/raw_data/`.
-   `<num_future_timesteps>`: The number of future timesteps the network should predict.

**Options:**

-   `--num-past-timesteps <n>`: The number of past timesteps to predict (default: 0).
-   `--lr <learning_rate>`: Learning rate for the optimizer (default: 0.0007).
-   `--batch-size <size>`: Batch size for training (default: 32).
-   `--num-epochs <epochs>`: Number of training epochs (default: 100).

The script will automatically:
1.  Locate the raw data files for the specified policy.
2.  Segregate the simulation files into training, validation, and test sets to prevent data leakage.
3.  Preprocess the data for each set.
4.  Train the `DoxascopeNet` model, using early stopping based on validation accuracy.
5.  Save the best model and generate analysis plots.

### 3. Analysis

The training script automatically runs the analysis at the end of training. The output includes:
-   `training_curves.png`: Shows the training and validation loss and accuracy over epochs. The reported validation accuracy is the average across all predicted timesteps.
-   `multistep_accuracy.png`: Plots the model's final test accuracy for each predicted timestep. The x-axis shows the timestep relative to the present (e.g., `t-1`, `t+1`, `t+5`), providing a clear view of how predictability changes over time.
-   `confusion_matrix_t-k.png` (if applicable): A confusion matrix for the furthest *past* timestep predicted (e.g., `t-20`), showing how well the model identifies previous actions.
-   `confusion_matrix_t+k.png` (if applicable): A confusion matrix for the furthest *future* timestep predicted (e.g., `t+20`), allowing for analysis of long-term predictive patterns.

### 4. Hyperparameter Sweep

To find the best hyperparameters or network architecture, you can use the `doxascope_sweep.py` script.

**Usage:**

```bash
python -m doxascope.doxascope_sweep <policy_name> <num_future_timesteps> [options]
```

**Arguments:**

-   `<policy_name>`: The name of the policy to sweep.
-   `<num_future_timesteps>`: The number of future steps to predict.

**Options:**

-   `--num-configs <n>`: Number of random configurations to test (default: 30).
-   `--max-epochs <n>`: Maximum number of epochs for each run (default: 50).
-   `--sweep-type <type>`: Type of sweep: `hyper` (default) or `arch`.

The script will save the sweep results to a JSON file in the results directory.

### 5. Standalone Analysis

The `doxascope_analysis.py` script provides more in-depth analysis tools.

**Usage:**

```bash
python -m doxascope.doxascope_analysis <command> [args]
```

**Commands:**

-   `inspect <policy_name>`: Inspects the preprocessed data for a policy.
-   `encoding <policy_name>`: Analyzes how a trained model encodes information by running the test data through the network and visualizing the internal representations.
-   `sweep <results_path>`: Analyzes sweep results from a JSON file.

Run with `-h` or `--help` for more details on each command.
