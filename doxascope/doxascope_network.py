#!/usr/bin/env python3
"""
Doxascope Neural Network

A PyTorch implementation of a neural network that predicts agent movement from LSTM memory vectors.
"""

import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .doxascope_data import preprocess_doxascope_data


def get_activation_fn(name: str):
    """Returns an activation function module based on its name."""
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    return nn.GELU()  # Default to GELU


class DoxascopeDataset(Dataset):
    """Dataset for doxascope training."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DoxascopeNet(nn.Module):
    """
    Doxascope Neural Network Architecture

    """

    def __init__(
        self,
        input_dim=512,
        hidden_dim=512,
        num_classes=5,
        dropout_rate=0.4,
        num_future_timesteps=1,
        activation_fn="silu",
        main_net_depth=3,
        processor_depth=1,
        skip_connection_weight=0.1,
        num_past_timesteps=0,
    ):
        super(DoxascopeNet, self).__init__()

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "num_future_timesteps": num_future_timesteps,
            "num_past_timesteps": num_past_timesteps,
            "activation_fn": activation_fn,
            "main_net_depth": main_net_depth,
            "processor_depth": processor_depth,
            "skip_connection_weight": skip_connection_weight,
        }

        act_fn = get_activation_fn(activation_fn)
        lstm_state_dim = input_dim // 2
        processor_output_dim = hidden_dim // 2

        def build_mlp(depth, in_dim, out_dim):
            layers = []
            for i in range(depth):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout_rate))
                in_dim = out_dim
            return nn.Sequential(*layers)

        self.hidden_processor = build_mlp(processor_depth, lstm_state_dim, processor_output_dim)
        self.cell_processor = build_mlp(processor_depth, lstm_state_dim, processor_output_dim)

        main_net_layers = []
        main_in_dim = hidden_dim  # from concatenated processors
        if main_net_depth > 0:
            main_out_dim = hidden_dim // 2
            # Input layer
            main_net_layers.extend([nn.Linear(main_in_dim, hidden_dim), act_fn, nn.Dropout(dropout_rate)])
            # Hidden layers
            for _ in range(main_net_depth - 2):
                main_net_layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Dropout(dropout_rate)])
            # Output layer
            if main_net_depth > 1:
                main_net_layers.extend([nn.Linear(hidden_dim, main_out_dim), act_fn, nn.Dropout(dropout_rate)])
            head_input_dim = main_out_dim
        else:
            head_input_dim = main_in_dim

        self.main_net = nn.Sequential(*main_net_layers)

        num_heads = num_past_timesteps + num_future_timesteps
        self.output_heads = nn.ModuleList([nn.Linear(head_input_dim, num_classes) for _ in range(num_heads)])
        self.skip_connection = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Input x is expected to be a concatenation of the LSTM hidden state and cell state.
        lstm_state_dim = self.config["input_dim"] // 2

        # Split input into hidden and cell states
        hidden_state = x[:, :lstm_state_dim]
        cell_state = x[:, lstm_state_dim:]

        # Process each state through its respective MLP
        h_processed = self.hidden_processor(hidden_state)
        c_processed = self.cell_processor(cell_state)

        # Combine the processed features
        combined = torch.cat([h_processed, c_processed], dim=1)

        # Pass through the main processing network
        main_features = self.main_net(combined)

        # Generate predictions from each output head
        outputs = [head(main_features) for head in self.output_heads]

        # Apply a residual skip connection to the first future timestep's prediction
        skip_weight = self.config.get("skip_connection_weight", 0.0)
        if skip_weight > 0 and self.config["num_future_timesteps"] > 0:
            skip_output = self.skip_connection(x)
            first_future_idx = self.config["num_past_timesteps"]
            if first_future_idx < len(outputs):
                outputs[first_future_idx] = outputs[first_future_idx] + skip_weight * skip_output

        return outputs


class DoxascopeTrainer:
    """Training and evaluation pipeline for the doxascope network."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def _compute_loss(self, outputs: List[torch.Tensor], batch_y: torch.Tensor) -> torch.Tensor:
        """Computes the total loss across all prediction heads."""
        loss = torch.tensor(0.0, device=self.device)
        for i, out in enumerate(outputs):
            target = batch_y[:, i]
            loss += self.criterion(out, target)
        return loss

    def _run_epoch(self, dataloader: DataLoader, is_training: bool):
        """Run a single epoch of training or evaluation."""
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_samples = 0
        num_steps = self.model.config.get("num_past_timesteps", 0) + self.model.config.get("num_future_timesteps", 0)
        correct_per_step = [0] * num_steps
        all_preds_per_step = [[] for _ in range(num_steps)]
        all_targets_per_step = [[] for _ in range(num_steps)]

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self._compute_loss(outputs, batch_y)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * batch_y.size(0)
                total_samples += batch_y.size(0)

                for i in range(num_steps):
                    _, predicted = outputs[i].max(1)
                    correct_per_step[i] += predicted.eq(batch_y[:, i]).sum().item()
                    if not is_training:
                        all_preds_per_step[i].extend(predicted.cpu().numpy())
                        all_targets_per_step[i].extend(batch_y[:, i].cpu().numpy())

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        acc_per_step = [100.0 * c / total_samples for c in correct_per_step] if total_samples > 0 else [0.0] * num_steps

        return avg_loss, acc_per_step, all_preds_per_step, all_targets_per_step

    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10):
        """Train the model and return training history and results."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0
        epochs_no_improve = 0

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_acc_per_step": []}

        checkpoint = {}

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc_per_step, _, _ = self._run_epoch(train_loader, is_training=True)
            val_loss, val_acc_per_step, _, _ = self._run_epoch(val_loader, is_training=False)

            # Average accuracy across all prediction steps
            avg_train_acc = sum(train_acc_per_step) / len(train_acc_per_step) if train_acc_per_step else 0
            avg_val_acc = sum(val_acc_per_step) / len(val_acc_per_step) if val_acc_per_step else 0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_acc"].append(avg_val_acc)
            history["val_acc_per_step"].append(val_acc_per_step)

            epoch_duration = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs} - {epoch_duration:.2f}s - "
                f"Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                f"Acc: {avg_train_acc:.2f}% - Val Acc: {avg_val_acc:.2f}%"
            )

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                epochs_no_improve = 0
                # Save best model checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.model.config,
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

        # Load best model for final evaluation
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])

        # Final evaluation on validation set to get predictions
        _, test_acc_per_step, all_preds, all_targets = self._run_epoch(val_loader, is_training=False)

        results = {
            "test_acc_per_step": test_acc_per_step,
            "predictions": all_preds,
            "targets": all_targets,
            "model_config": self.model.config,
        }

        return history, results, checkpoint


def prepare_data(
    raw_data_dir: Path,
    output_dir: Path,
    test_split: float,
    val_split: float,
    num_future_timesteps: int,
    num_past_timesteps: int,
):
    """
    Prepares and splits data into training, validation, and test sets.
    """
    preprocessed_dir = output_dir / "preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    all_json_files = sorted(list(raw_data_dir.glob("*.json")))
    if not all_json_files:
        print(f"No JSON files found in {raw_data_dir}")
        return None, None

    # Split files before preprocessing to prevent data leakage
    num_files = len(all_json_files)
    test_idx = int(num_files * (1 - test_split))
    val_idx = int(test_idx * (1 - val_split))

    train_files = all_json_files[:val_idx]
    val_files = all_json_files[val_idx:test_idx]
    test_files = all_json_files[test_idx:]

    print(f"Splitting {num_files} files: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Preprocess each split
    X_train, y_train = preprocess_doxascope_data(
        train_files, preprocessed_dir, "train_data.npz", num_future_timesteps, num_past_timesteps
    )
    X_val, y_val = preprocess_doxascope_data(
        val_files, preprocessed_dir, "val_data.npz", num_future_timesteps, num_past_timesteps
    )
    X_test, y_test = preprocess_doxascope_data(
        test_files, preprocessed_dir, "test_data.npz", num_future_timesteps, num_past_timesteps
    )

    if X_train is None or y_train is None:
        print("Training data could not be created.")
        return None, None

    input_dim = X_train.shape[1]

    train_loader = DataLoader(DoxascopeDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(DoxascopeDataset(X_val, y_val), batch_size=32, shuffle=False) if X_val is not None else None
    test_loader = (
        DataLoader(DoxascopeDataset(X_test, y_test), batch_size=32, shuffle=False) if X_test is not None else None
    )

    return (train_loader, val_loader, test_loader), input_dim
