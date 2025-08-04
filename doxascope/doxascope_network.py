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


def create_baseline_data(preprocessed_dir: Path, batch_size: int) -> tuple:
    """Creates baseline data loaders by loading preprocessed data and randomizing inputs."""
    baseline_files = {
        "train": preprocessed_dir / "train_baseline.npz",
        "val": preprocessed_dir / "val_baseline.npz",
        "test": preprocessed_dir / "test_baseline.npz",
    }

    # Check if baseline files already exist
    if all(f.exists() for f in baseline_files.values()):
        print("Loading existing baseline data files...")
    else:
        print("Creating baseline data files...")
        # Load original preprocessed data
        original_files = {
            "train": preprocessed_dir / "train.npz",
            "val": preprocessed_dir / "val.npz",
            "test": preprocessed_dir / "test.npz",
        }

        for split, baseline_file in baseline_files.items():
            original_file = original_files[split]
            if original_file.exists():
                data = np.load(original_file)
                X, y = data["X"], data["y"]
                X_random = np.random.randn(*X.shape).astype(np.float32)
                np.savez_compressed(baseline_file, X=X_random, y=y)

    # Create data loaders from baseline files
    loaders = []
    for split in ["train", "val", "test"]:
        baseline_file = baseline_files[split]
        if baseline_file.exists():
            data = np.load(baseline_file)
            dataset = DoxascopeDataset(data["X"], data["y"])
            shuffle = split == "train"
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            loaders.append(loader)
            print(f"  Baseline {split} samples: {len(dataset)}")
        else:
            loaders.append(None)

    # Get input dimension from first available dataset
    input_dim = None
    for loader in loaders:
        if loader is not None:
            sample_x, _ = next(iter(loader))
            input_dim = sample_x.shape[1]
            break

    return tuple(loaders) + (input_dim,)


def prepare_data(
    raw_data_dir: Path,
    output_dir: Path,
    batch_size: int,
    test_split: float,
    val_split: float,
    num_future_timesteps: int,
    num_past_timesteps: int,
    data_split_seed: int = 42,
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

    # Shuffle files deterministically to ensure reproducible splits
    import random

    random.seed(data_split_seed)
    random.shuffle(all_json_files)

    # Create more balanced splits by distributing files more evenly
    # This helps ensure similar label distributions across splits
    num_files = len(all_json_files)

    # Calculate split indices
    test_idx = max(1, int(num_files * test_split))
    val_idx = test_idx + max(1, int(num_files * val_split))

    # Clamp indices to ensure valid splits
    test_idx = min(test_idx, num_files)
    val_idx = min(val_idx, num_files)

    # Balance file distribution by file size to get more even label distributions
    # while maintaining file-level separation to prevent data leakage

    # Get file sizes to help balance the splits
    file_sizes = []
    for file in all_json_files:
        size = file.stat().st_size
        file_sizes.append((file, size))

    # Sort by file size for better distribution
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # Distribute files across splits to balance total data volume
    # Use a round-robin approach weighted by target split sizes
    test_files = []
    val_files = []
    train_files = []

    # Target ratios
    test_ratio = test_split
    val_ratio = val_split
    train_ratio = 1.0 - test_split - val_split

    total_size = sum(size for _, size in file_sizes)
    target_test_size = total_size * test_ratio
    target_val_size = total_size * val_ratio

    current_test_size = 0
    current_val_size = 0

    # Distribute files to balance data volume across splits
    for file, size in file_sizes:
        test_need = target_test_size - current_test_size
        val_need = target_val_size - current_val_size

        # Assign to split with highest need
        if test_need > val_need and test_need > 0:
            test_files.append(file)
            current_test_size += size
        elif val_need > 0:
            val_files.append(file)
            current_val_size += size
        else:
            train_files.append(file)

    print(f"Data split into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.")
    print(
        f"File size distribution - Train: {sum(f.stat().st_size for f in train_files) / 1024000:.1f}MB, "
        f"Val: {sum(f.stat().st_size for f in val_files) / 1024000:.1f}MB, "
        f"Test: {sum(f.stat().st_size for f in test_files) / 1024000:.1f}MB"
    )

    # Process files for each split
    X_train, y_train = preprocess_doxascope_data(
        train_files, preprocessed_dir, "train.npz", num_future_timesteps, num_past_timesteps
    )
    if X_train is None:
        print("No training data could be generated.")
        return None, None, None, None
    input_dim = X_train.shape[1]

    X_val, y_val = preprocess_doxascope_data(
        val_files, preprocessed_dir, "val.npz", num_future_timesteps, num_past_timesteps
    )
    if X_val is None:
        print("Warning: No validation samples could be generated from the validation files.")

    X_test, y_test = preprocess_doxascope_data(
        test_files, preprocessed_dir, "test.npz", num_future_timesteps, num_past_timesteps
    )
    if X_test is None:
        print("Warning: No test samples could be generated from the test files.")

    # Create datasets and dataloaders
    train_dataset = DoxascopeDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"  Train samples: {len(train_loader.dataset)}")

    if X_val is not None and y_val is not None:
        val_dataset = DoxascopeDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"  Validation samples: {len(val_loader.dataset)}")

    if X_test is not None and y_test is not None:
        test_dataset = DoxascopeDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"  Test samples: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, input_dim
