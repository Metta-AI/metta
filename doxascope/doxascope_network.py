#!/usr/bin/env python3
"""
Doxascope Neural Network

A PyTorch implementation of a neural network that predicts agent movement from LSTM memory vectors.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .doxascope_data import get_num_classes_for_manhattan_distance, preprocess_doxascope_data


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


@dataclass
class TrainingResult:
    """Structured results from a training run."""

    history: Dict[str, List[float]]
    best_checkpoint: Dict
    final_val_acc: float


class DoxascopeNet(nn.Module):
    """
    Doxascope Neural Network Architecture

    """

    def __init__(
        self,
        input_dim=512,
        hidden_dim=512,
        dropout_rate=0.4,
        num_future_timesteps: int = 1,
        num_past_timesteps: int = 0,
        activation_fn="silu",
        main_net_depth=3,
        processor_depth=1,
    ):
        super(DoxascopeNet, self).__init__()

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "num_future_timesteps": num_future_timesteps,
            "num_past_timesteps": num_past_timesteps,
            "activation_fn": activation_fn,
            "main_net_depth": main_net_depth,
            "processor_depth": processor_depth,
        }

        act_fn = get_activation_fn(activation_fn)
        lstm_state_dim = input_dim // 2
        processor_output_dim = hidden_dim // 2

        def build_mlp(depth, in_dim, h_dim, out_dim):
            layers = []
            if depth == 0:
                return nn.Identity()
            if depth == 1:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout_rate))
                for _ in range(depth - 2):
                    layers.append(nn.Linear(h_dim, h_dim))
                    layers.append(act_fn)
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.Linear(h_dim, out_dim))
            return nn.Sequential(*layers)

        self.hidden_processor = build_mlp(processor_depth, lstm_state_dim, hidden_dim, processor_output_dim)
        self.cell_processor = build_mlp(processor_depth, lstm_state_dim, hidden_dim, processor_output_dim)

        main_in_dim = processor_output_dim * 2
        self.main_net = build_mlp(main_net_depth, main_in_dim, hidden_dim, hidden_dim)

        # Create heads for past and future predictions
        self.output_heads = nn.ModuleList()
        self.head_timesteps = sorted(
            [k for k in range(-num_past_timesteps, 0)] + [k for k in range(1, num_future_timesteps + 1)]
        )

        for k in self.head_timesteps:
            num_classes = get_num_classes_for_manhattan_distance(abs(k))
            self.output_heads.append(nn.Linear(hidden_dim, num_classes))

        # No skip connection for this classification model yet
        # self.skip_connection = nn.Linear(input_dim, output_dim)

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
        return loss / len(outputs) if outputs else loss

    def _run_epoch(self, dataloader: DataLoader, is_training: bool):
        """Run a single epoch of training or evaluation."""
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_samples = 0
        num_steps = len(self.model.output_heads)
        correct_per_step = [0] * num_steps

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

                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

                for i in range(num_steps):
                    _, predicted = outputs[i].max(1)
                    correct_per_step[i] += predicted.eq(batch_y[:, i]).sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        acc_per_step = (
            [100.0 * c / total_samples for c in correct_per_step] if total_samples > 0 else ([0.0] * num_steps)
        )

        return avg_loss, acc_per_step

    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10) -> Optional[TrainingResult]:
        """Train the model and return training history and results."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0
        epochs_no_improve = 0

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        checkpoint = {}

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc_per_step = self._run_epoch(train_loader, is_training=True)
            avg_train_acc = sum(train_acc_per_step) / len(train_acc_per_step) if train_acc_per_step else 0

            val_loss, val_acc_per_step = (
                self._run_epoch(val_loader, is_training=False) if val_loader else (float("inf"), [])
            )
            avg_val_acc = sum(val_acc_per_step) / len(val_acc_per_step) if val_acc_per_step else 0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_acc"].append(avg_val_acc)

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
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # If no checkpoint was ever saved (e.g., no val set), save the final model state
        if not checkpoint:
            print("Saving model from the final epoch as no best model was found.")
            checkpoint = {
                "epoch": num_epochs,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.model.config,
            }

        return TrainingResult(history=history, best_checkpoint=checkpoint, final_val_acc=best_val_acc)


def prepare_data(
    raw_data_dir: Path,
    output_dir: Path,
    test_split: float,
    val_split: float,
    num_future_timesteps: int,
    num_past_timesteps: int,
    randomize_X: bool = False,
):
    """
    Prepares and splits data into training, validation, and test sets.
    The split is done on a per-file basis to prevent data leakage.
    """
    train_loader, val_loader, test_loader = None, None, None
    preprocessed_dir = output_dir / "preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    all_json_files = sorted(list(raw_data_dir.glob("*.json")))
    if not all_json_files:
        raise ValueError(f"No JSON files found in {raw_data_dir}")

    # Shuffle and split the files to prevent data leakage
    import random

    random.shuffle(all_json_files)

    num_files = len(all_json_files)
    test_idx = int(num_files * test_split)
    val_idx = test_idx + int(num_files * val_split)

    # Ensure there's at least one file for training if splits are small
    if num_files > 2 and val_idx == test_idx:
        val_idx = test_idx + 1
    if num_files > 1 and test_idx == 0:
        test_idx = 1
    # Ensure train set is not empty if we have enough files
    if num_files > 2 and val_idx == num_files:
        val_idx = num_files - 1

    test_files = all_json_files[:test_idx]
    val_files = all_json_files[test_idx:val_idx]
    train_files = all_json_files[val_idx:]

    print(f"Data split into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.")

    # Process files for each split
    print(f"Processing {len(train_files)} files for training...")
    X_train, y_train = preprocess_doxascope_data(
        train_files, preprocessed_dir, "train.npz", num_future_timesteps, num_past_timesteps
    )
    if X_train is None:
        print("No training data could be generated.")
        return None, None, None, None

    print(f"Processing {len(val_files)} files for validation...")
    X_val, y_val = preprocess_doxascope_data(
        val_files, preprocessed_dir, "val.npz", num_future_timesteps, num_past_timesteps
    )
    if X_val is None:
        print("Warning: No validation samples could be generated from the validation files.")

    print(f"Processing {len(test_files)} files for testing...")
    X_test, y_test = preprocess_doxascope_data(
        test_files, preprocessed_dir, "test.npz", num_future_timesteps, num_past_timesteps
    )
    if X_test is None:
        print("Warning: No test samples could be generated from the test files.")

    if val_loader:
        print(f"  Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"  Test samples: {len(test_loader.dataset)}")

    input_dim = X_train.shape[1]

    if randomize_X:
        print("Randomizing input features for baseline training.")
        if isinstance(X_train, np.ndarray):
            print(f"Original X_train mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
            X_train[:] = np.random.randn(*X_train.shape)
            print(f"Randomized X_train mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
        if X_val is not None and isinstance(X_val, np.ndarray):
            X_val[:] = np.random.randn(*X_val.shape)
        if X_test is not None and isinstance(X_test, np.ndarray):
            X_test[:] = np.random.randn(*X_test.shape)

    # Create datasets and dataloaders
    train_dataset = DoxascopeDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = DoxascopeDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = DoxascopeDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim
