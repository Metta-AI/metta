"""
Doxascope Neural Network

A PyTorch implementation of a neural network that predicts agent movement from LSTM memory vectors.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .doxascope_data import (
    find_changing_items_across_files,
    get_num_classes_for_manhattan_distance,
    get_num_classes_for_quadrant_granularity,
    preprocess_doxascope_data,
)


def get_activation_fn(name: str) -> nn.Module:
    """Returns an activation function module based on its name."""
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    return nn.GELU()  # Default to GELU


class DoxascopeDataset(Dataset):
    """Dataset for doxascope training with location and optional inventory labels."""

    def __init__(
        self,
        X: np.ndarray,
        y_location: np.ndarray,
        y_inventory: Optional[np.ndarray] = None,
        time_to_change: Optional[np.ndarray] = None,
    ):
        self.X = torch.FloatTensor(X)
        self.y_location = torch.LongTensor(y_location)
        # y_inventory is now class indices (LongTensor), not multi-hot vectors
        self.y_inventory = torch.LongTensor(y_inventory) if y_inventory is not None else None
        self.time_to_change = torch.LongTensor(time_to_change) if time_to_change is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "X": self.X[idx],
            "y_location": self.y_location[idx],
        }
        if self.y_inventory is not None:
            item["y_inventory"] = self.y_inventory[idx]
        if self.time_to_change is not None:
            item["time_to_change"] = self.time_to_change[idx]
        return item


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
        input_dim: int,
        num_future_timesteps: int,
        num_past_timesteps: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.4,
        activation_fn: str = "gelu",
        main_net_depth: int = 3,
        processor_depth: int = 1,
        granularity: str = "exact",
        prediction_types: Optional[List[str]] = None,
        inventory_num_items: Optional[int] = None,
    ):
        super(DoxascopeNet, self).__init__()

        # Set default prediction types
        if prediction_types is None:
            prediction_types = ["location"]
        self.prediction_types = prediction_types

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "num_future_timesteps": num_future_timesteps,
            "num_past_timesteps": num_past_timesteps,
            "activation_fn": activation_fn,
            "main_net_depth": main_net_depth,
            "processor_depth": processor_depth,
            "granularity": granularity,
            "prediction_types": prediction_types,
        }

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
                layers.append(get_activation_fn(activation_fn))  # Create new instance for each use
                layers.append(nn.Dropout(dropout_rate))
                for _ in range(depth - 2):
                    layers.append(nn.Linear(h_dim, h_dim))
                    layers.append(get_activation_fn(activation_fn))  # Create new instance for each use
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.Linear(h_dim, out_dim))
            return nn.Sequential(*layers)

        self.hidden_processor = build_mlp(processor_depth, lstm_state_dim, hidden_dim, processor_output_dim)
        self.cell_processor = build_mlp(processor_depth, lstm_state_dim, hidden_dim, processor_output_dim)

        main_in_dim = processor_output_dim * 2
        self.main_net = build_mlp(main_net_depth, main_in_dim, hidden_dim, hidden_dim)

        # Create unified prediction specs list
        prediction_specs: List[Tuple[str, Optional[int]]] = []

        # Location predictions at various timesteps (only if location is enabled)
        if "location" in prediction_types:
            self.head_timesteps = sorted(
                [k for k in range(-num_past_timesteps, 0)] + [k for k in range(1, num_future_timesteps + 1)]
            )
            for k in self.head_timesteps:
                prediction_specs.append(("location", k))
        else:
            self.head_timesteps = []

        # Inventory prediction (next inventory change)
        if "inventory" in prediction_types:
            if inventory_num_items is None:
                raise ValueError("inventory_num_items must be provided when 'inventory' is in prediction_types")
            prediction_specs.append(("inventory", None))
            self.config["inventory_num_items"] = inventory_num_items

        self.prediction_specs = prediction_specs

        # Unified conditioning via FiLM (Feature-wise Linear Modulation)
        # Single embedding table for all prediction types
        prediction_embedding_dim = 32
        num_prediction_specs = len(prediction_specs)
        self.prediction_embedding = nn.Embedding(num_prediction_specs, prediction_embedding_dim)
        self.prediction_processor = nn.Linear(prediction_embedding_dim, hidden_dim * 2)

        # Create mapping from prediction spec to embedding index
        self.prediction_to_idx = {spec: i for i, spec in enumerate(prediction_specs)}

        # Output adapters: one per prediction type
        self.output_adapters = nn.ModuleDict()

        # Location adapter
        if "location" in prediction_types:
            # Compute maximum number of classes across all location timesteps
            max_num_classes = 0
            for k in self.head_timesteps:
                if granularity == "exact":
                    num_classes = get_num_classes_for_manhattan_distance(abs(k))
                elif granularity == "quadrant":
                    num_classes = get_num_classes_for_quadrant_granularity(abs(k))
                else:
                    raise ValueError(f"Unknown granularity: {granularity}")
                max_num_classes = max(max_num_classes, num_classes)

            self.output_adapters["location"] = nn.Linear(hidden_dim, max_num_classes)
            self.config["max_num_classes"] = max_num_classes

            # Create per-timestep class masks for location predictions
            self.location_masks: Dict[int, torch.Tensor] = {}
            for k in self.head_timesteps:
                if granularity == "exact":
                    num_classes = get_num_classes_for_manhattan_distance(abs(k))
                elif granularity == "quadrant":
                    num_classes = get_num_classes_for_quadrant_granularity(abs(k))
                else:
                    raise ValueError(f"Unknown granularity: {granularity}")
                # Create boolean mask: True for valid classes, False for invalid
                mask = torch.zeros(max_num_classes, dtype=torch.bool)
                mask[:num_classes] = True
                self.register_buffer(f"location_mask_k{k}", mask)
                self.location_masks[k] = mask

        # Inventory adapter
        if "inventory" in prediction_types:
            # Multi-label binary classification: which items changed
            # Output: [batch, inventory_num_items] with sigmoid → probabilities
            self.output_adapters["inventory"] = nn.Linear(hidden_dim, inventory_num_items)

        # Store embedding dim and architecture type in config
        self.config["prediction_embedding_dim"] = prediction_embedding_dim
        self.config["architecture"] = "unified_conditioned_head"

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Input x is expected to be a concatenation of the LSTM hidden state and cell state.
        assert x.shape[1] % 2 == 0, "Input dimension must be even (equal hidden and cell state sizes)"
        lstm_state_dim = x.shape[1] // 2

        # Split input into hidden and cell states
        hidden_state = x[:, :lstm_state_dim]
        cell_state = x[:, lstm_state_dim:]

        # Process each state through its respective MLP
        h_processed = self.hidden_processor(hidden_state)
        c_processed = self.cell_processor(cell_state)

        # Combine the processed features
        combined = torch.cat([h_processed, c_processed], dim=1)

        # Pass through the main processing network
        main_features = self.main_net(combined)  # [batch, hidden_dim]

        # Generate predictions for each prediction spec using unified conditioning
        outputs = []
        batch_size = x.size(0)
        device = x.device

        for prediction_spec in self.prediction_specs:
            task_type, timestep = prediction_spec

            # Get prediction spec index for embedding lookup
            spec_idx = self.prediction_to_idx[prediction_spec]
            spec_idx_tensor = torch.full((batch_size,), spec_idx, device=device, dtype=torch.long)

            # Embed prediction spec
            spec_emb = self.prediction_embedding(spec_idx_tensor)  # [batch, prediction_embedding_dim]

            # Generate FiLM parameters (scale and shift)
            film_params = self.prediction_processor(spec_emb)  # [batch, hidden_dim * 2]
            scale = film_params[:, : self.config["hidden_dim"]]
            shift = film_params[:, self.config["hidden_dim"] :]

            # Apply FiLM conditioning: modulate features element-wise
            conditioned_features = main_features * (1 + scale) + shift  # [batch, hidden_dim]

            # Get predictions from task-specific adapter
            adapter = self.output_adapters[task_type]
            logits = adapter(conditioned_features)

            # Apply task-specific post-processing
            if task_type == "location":
                # Apply masking: set invalid classes to -inf so they're ignored in softmax/argmax
                mask = self.location_masks[timestep].to(device)
                logits = logits.masked_fill(~mask, float("-inf"))
            elif task_type == "inventory":
                # No masking needed for inventory (all outputs are valid)
                # Logits will be used with BCEWithLogitsLoss (sigmoid applied in loss)
                pass

            outputs.append(logits)

        return outputs


@dataclass
class EpochMetrics:
    """Metrics from a single training/evaluation epoch."""

    avg_loss: float
    location_acc_per_step: List[float]
    inventory_accuracy: Optional[float] = None
    inventory_per_item_accuracy: Optional[List[float]] = None


class DoxascopeTrainer:
    """Training and evaluation pipeline for the doxascope network."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def _compute_loss(self, outputs: List[torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the total loss across all prediction heads."""
        loss = torch.tensor(0.0, device=self.device)
        num_heads = 0

        # Location losses (CrossEntropy for each timestep)
        num_location_heads = len(self.model.head_timesteps)
        if "location" in self.model.prediction_types and "y_location" in batch:
            ce_criterion = nn.CrossEntropyLoss()
            y_location = batch["y_location"]
            for i in range(num_location_heads):
                target = y_location[:, i]
                loss += ce_criterion(outputs[i], target)
            num_heads += num_location_heads

        # Inventory loss (CrossEntropy - single class prediction, not multi-label)
        if "inventory" in self.model.prediction_types and "y_inventory" in batch:
            inv_output = outputs[num_location_heads]  # Inventory is last output
            y_inventory = batch["y_inventory"]  # Class indices [batch_size]
            ce_criterion = nn.CrossEntropyLoss()
            loss += ce_criterion(inv_output, y_inventory)
            num_heads += 1

        return loss / num_heads if num_heads > 0 else loss

    def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> EpochMetrics:
        """Run a single epoch of training or evaluation."""
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        num_location_steps = len(self.model.head_timesteps)
        correct_per_step = [0] * num_location_steps

        # Inventory tracking
        has_inventory = "inventory" in self.model.prediction_types
        inv_correct_total = 0
        inv_total = 0
        inv_correct_per_item: Optional[List[int]] = None
        inv_total_per_item: Optional[List[int]] = None
        if has_inventory:
            num_items = self.model.config.get("inventory_num_items", 0)
            inv_correct_per_item = [0] * num_items
            inv_total_per_item = [0] * num_items

        has_location = "location" in self.model.prediction_types

        for batch in dataloader:
            batch_x = batch["X"].to(self.device)

            # Move all batch items to device
            batch_on_device: Dict[str, torch.Tensor] = {}
            if has_location and "y_location" in batch:
                batch_on_device["y_location"] = batch["y_location"].to(self.device)
            if has_inventory and "y_inventory" in batch:
                batch_on_device["y_inventory"] = batch["y_inventory"].to(self.device)

            # Forward pass
            if not is_training:
                with torch.no_grad():
                    outputs = self.model(batch_x)
                    loss = self._compute_loss(outputs, batch_on_device)
            else:
                outputs = self.model(batch_x)
                loss = self._compute_loss(outputs, batch_on_device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute accuracy metrics
            with torch.no_grad():
                # Location accuracy
                if has_location and "y_location" in batch_on_device:
                    for i in range(num_location_steps):
                        _, predicted = outputs[i].max(1)
                        correct_per_step[i] += predicted.eq(batch_on_device["y_location"][:, i]).sum().item()

                # Inventory accuracy (single-class prediction)
                if has_inventory and "y_inventory" in batch_on_device:
                    inv_output = outputs[num_location_steps]
                    inv_pred = inv_output.argmax(dim=1)  # Predicted class
                    inv_target = batch_on_device["y_inventory"]  # True class

                    # Overall accuracy
                    correct_samples = (inv_pred == inv_target).sum().item()
                    inv_correct_total += correct_samples
                    inv_total += batch_size

                    # Per-item accuracy (accuracy when predicting each item class)
                    for j in range(len(inv_correct_per_item)):
                        mask = inv_target == j
                        if mask.any():
                            inv_correct_per_item[j] += (inv_pred[mask] == j).sum().item()
                            inv_total_per_item[j] += mask.sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        location_acc = (
            [100.0 * c / total_samples for c in correct_per_step] if total_samples > 0 else [0.0] * num_location_steps
        )

        # Compute inventory metrics
        inventory_accuracy = None
        inventory_per_item_accuracy = None
        if has_inventory and inv_total > 0:
            inventory_accuracy = 100.0 * inv_correct_total / inv_total
            inventory_per_item_accuracy = [
                100.0 * inv_correct_per_item[j] / inv_total_per_item[j] if inv_total_per_item[j] > 0 else 0.0
                for j in range(len(inv_correct_per_item))
            ]

        return EpochMetrics(
            avg_loss=avg_loss,
            location_acc_per_step=location_acc,
            inventory_accuracy=inventory_accuracy,
            inventory_per_item_accuracy=inventory_per_item_accuracy,
        )

    def train_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        num_epochs: int,
        lr: float,
        patience: int,
        output_dir: Path,
        policy_name: str,
        is_baseline: bool = False,
        resource_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Runs the full training and evaluation pipeline.
        """
        training_result = self.train(train_loader, val_loader, num_epochs, lr, patience)

        if training_result is None:
            print(f"🏁 Training finished early for {'baseline' if is_baseline else 'main'} model with no improvement.")
            return None

        # --- Evaluation on Test Set ---
        test_metrics: Optional[EpochMetrics] = None
        if test_loader:
            print("\n--- Evaluating on Test Set ---")
            self.model.load_state_dict(training_result.best_checkpoint["state_dict"])
            test_metrics = self._run_epoch(test_loader, is_training=False)
            timesteps = self.model.head_timesteps
            test_acc_per_step = test_metrics.location_acc_per_step
            test_acc_avg = sum(test_acc_per_step) / len(test_acc_per_step) if test_acc_per_step else 0.0

            print(f"  - Average Location Accuracy: {test_acc_avg:.2f}%")
            for step, acc in zip(timesteps, test_acc_per_step, strict=False):
                print(f"    - Timestep {step}: {acc:.2f}%")

            # Print inventory metrics if available
            if test_metrics.inventory_accuracy is not None:
                print(f"  - Inventory Prediction Accuracy: {test_metrics.inventory_accuracy:.2f}%")
                if test_metrics.inventory_per_item_accuracy and resource_names:
                    for name, acc in zip(resource_names, test_metrics.inventory_per_item_accuracy, strict=False):
                        print(f"    - {name}: {acc:.2f}%")
        else:
            test_acc_avg = 0.0
            timesteps = []
            test_acc_per_step = []

        # --- Save Results ---
        suffix = "_baseline" if is_baseline else ""
        test_results: Dict[str, Any] = {
            "policy_name": policy_name,
            "test_accuracy_avg": test_acc_avg,
            "timesteps": timesteps,
            "test_accuracy_per_step": test_acc_per_step,
            "model_config": self.model.config,
        }

        # Add inventory results if available
        if test_metrics and test_metrics.inventory_accuracy is not None:
            test_results["inventory_accuracy"] = {
                "overall": test_metrics.inventory_accuracy,
                "per_item": (
                    dict(zip(resource_names, test_metrics.inventory_per_item_accuracy, strict=False))
                    if resource_names and test_metrics.inventory_per_item_accuracy
                    else {}
                ),
            }

        with open(output_dir / f"test_results{suffix}.json", "w") as f:
            json.dump(test_results, f, indent=2)

        # Only save history and checkpoints for the main model
        if not is_baseline:
            history_df = pd.DataFrame(training_result.history)
            history_df.to_csv(output_dir / "training_history.csv", index=False)

            torch.save(training_result.best_checkpoint, output_dir / "best_model.pth")
            # Also save a state-dict-only file for robust cross-version loading in analysis
            try:
                torch.save(self.model.state_dict(), output_dir / "best_model.state_dict.pth")
            except Exception:
                pass
        print(f"✅ Results saved to {output_dir}")

        # Return artifacts needed for plotting
        return {
            "model": self.model,
            "history": training_result.history,
            "test_results": test_results,
            "test_loader": test_loader,
            "resource_names": resource_names,
        }

    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10) -> Optional[TrainingResult]:
        """Train the model and return training history and results."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0.0
        epochs_no_improve = 0

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_inv_acc": [],
            "val_inv_acc": [],
        }

        checkpoint = {}
        has_inventory = "inventory" in self.model.prediction_types

        for epoch in range(num_epochs):
            start_time = time.time()

            train_metrics = self._run_epoch(train_loader, is_training=True)
            avg_train_acc = (
                sum(train_metrics.location_acc_per_step) / len(train_metrics.location_acc_per_step)
                if train_metrics.location_acc_per_step
                else 0.0
            )

            if val_loader:
                val_metrics = self._run_epoch(val_loader, is_training=False)
                val_loss = val_metrics.avg_loss
                avg_val_acc = (
                    sum(val_metrics.location_acc_per_step) / len(val_metrics.location_acc_per_step)
                    if val_metrics.location_acc_per_step
                    else 0.0
                )
            else:
                val_metrics = None
                val_loss = float("inf")
                avg_val_acc = 0.0

            history["train_loss"].append(train_metrics.avg_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_acc"].append(avg_val_acc)
            history["train_inv_acc"].append(train_metrics.inventory_accuracy or 0.0)
            history["val_inv_acc"].append(val_metrics.inventory_accuracy if val_metrics else 0.0)

            epoch_duration = time.time() - start_time
            log_msg = (
                f"Epoch {epoch + 1}/{num_epochs} - {epoch_duration:.2f}s - "
                f"Loss: {train_metrics.avg_loss:.4f} - Val Loss: {val_loss:.4f} - "
                f"Loc Acc: {avg_train_acc:.2f}% - Val Loc Acc: {avg_val_acc:.2f}%"
            )
            if has_inventory:
                train_inv = train_metrics.inventory_accuracy or 0.0
                val_inv = val_metrics.inventory_accuracy if val_metrics else 0.0
                log_msg += f" - Inv Acc: {train_inv:.2f}% - Val Inv Acc: {val_inv:.2f}%"
            print(log_msg)

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


@dataclass
class PreparedData:
    """Container for prepared data loaders and metadata."""

    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    input_dim: Optional[int]
    resource_names: Optional[List[str]] = None
    inventory_num_items: Optional[int] = None


def create_baseline_data(preprocessed_dir: Path, batch_size: int) -> PreparedData:
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
                data = np.load(original_file, allow_pickle=True)
                X, y = data["X"], data["y"]
                X_random = np.asarray(np.random.randn(*X.shape), dtype=np.float32)

                # Preserve all keys from original file, just randomize X
                save_dict = {"X": X_random, "y": y}
                if "y_inventory" in data:
                    save_dict["y_inventory"] = data["y_inventory"]
                if "time_to_change" in data:
                    save_dict["time_to_change"] = data["time_to_change"]
                if "resource_names" in data:
                    save_dict["resource_names"] = data["resource_names"]
                if "granularity" in data:
                    save_dict["granularity"] = data["granularity"]

                np.savez_compressed(baseline_file, **save_dict)

    # Create data loaders from baseline files
    loaders: List[Optional[DataLoader]] = []
    resource_names = None
    for split in ["train", "val", "test"]:
        baseline_file = baseline_files[split]
        if baseline_file.exists():
            data = np.load(baseline_file, allow_pickle=True)
            y_inventory = data["y_inventory"] if "y_inventory" in data else None
            time_to_change = data["time_to_change"] if "time_to_change" in data else None
            dataset = DoxascopeDataset(data["X"], data["y"], y_inventory, time_to_change)

            if resource_names is None and "resource_names" in data:
                resource_names = list(data["resource_names"])

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
            sample = next(iter(loader))
            input_dim = sample["X"].shape[1]
            break

    inventory_num_items = len(resource_names) if resource_names else None

    return PreparedData(
        train_loader=loaders[0],
        val_loader=loaders[1],
        test_loader=loaders[2],
        input_dim=input_dim,
        resource_names=resource_names,
        inventory_num_items=inventory_num_items,
    )


def prepare_data(
    raw_data_dir: Path,
    output_dir: Path,
    batch_size: int,
    test_split: float,
    val_split: float,
    num_future_timesteps: int,
    num_past_timesteps: int,
    data_split_seed: int = 42,
    granularity: str = "exact",
    include_inventory: bool = True,
    exclude_inventory_items: Optional[List[str]] = None,
) -> PreparedData:
    """
    Prepares and splits data into training, validation, and test sets.
    The split is done on a per-file basis to prevent data leakage.

    Args:
        exclude_inventory_items: Optional list of item names to exclude from inventory prediction
            (e.g., ["energy"] to exclude passive resource consumption)
    """
    preprocessed_dir = output_dir / "preprocessed_data"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    all_json_files = sorted(list(raw_data_dir.glob("*.json")))
    if not all_json_files:
        raise ValueError(f"No JSON files found in {raw_data_dir}")

    import random

    random.seed(data_split_seed)
    random.shuffle(all_json_files)

    num_files = len(all_json_files)
    test_count = 0
    val_count = 0

    if num_files >= 3:
        test_count = int(round(num_files * test_split)) if test_split > 0 else 0
        val_count = int(round(num_files * val_split)) if val_split > 0 else 0

        if test_split > 0 and test_count == 0:
            test_count = 1
        if val_split > 0 and val_count == 0:
            val_count = 1

        reserved = test_count + val_count
        if num_files - reserved < 1:
            needed = 1 - (num_files - reserved)
            reduce_val = min(needed, val_count)
            val_count -= reduce_val
            needed -= reduce_val
            if needed > 0:
                reduce_test = min(needed, test_count)
                test_count -= reduce_test

        test_count = max(0, min(test_count, num_files))
        val_count = max(0, min(val_count, num_files - test_count))

    elif num_files == 2:
        if test_split > 0 or val_split > 0:
            test_count = 1
            val_count = 0

    # Now distribute round-robin after sorting:
    all_json_files.sort(key=lambda f: f.stat().st_size, reverse=True)

    train_files: List[Path] = []
    val_files: List[Path] = []
    test_files: List[Path] = []

    # List of (split_list, max_count) tuples, in assignment priority order
    active_splits = []
    if test_count > 0:
        active_splits.append((test_files, test_count))
    if val_count > 0:
        active_splits.append((val_files, val_count))
    train_max = num_files - test_count - val_count
    if train_max > 0:
        active_splits.append((train_files, train_max))

    assigned = 0
    while assigned < num_files and active_splits:
        for i in range(len(active_splits)):
            if assigned >= num_files:
                break
            split_list, remaining = active_splits[i]
            if remaining > 0:
                split_list.append(all_json_files[assigned])
                assigned += 1
                remaining -= 1
                active_splits[i] = (split_list, remaining)
                if remaining == 0:
                    del active_splits[i]
                    break  # Restart loop to avoid index issues

    train_files.extend(all_json_files[assigned:])

    print(f"Data split into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.")

    # Find changing items across ALL files first (ensures consistent item list across splits)
    resource_names = None
    if include_inventory:
        resource_names = find_changing_items_across_files(all_json_files, exclude_items=exclude_inventory_items)
        if exclude_inventory_items:
            print(f"Excluding inventory items from prediction: {exclude_inventory_items}")
        if resource_names:
            print(f"Found {len(resource_names)} items that change across all files: {resource_names}")
        else:
            print("No items change in any file. Disabling inventory prediction.")
            include_inventory = False

    # Process files for each split (using consistent resource_names)
    train_result = preprocess_doxascope_data(
        train_files,
        preprocessed_dir,
        "train.npz",
        num_future_timesteps,
        num_past_timesteps,
        granularity=granularity,
        include_inventory=include_inventory,
        resource_names_override=resource_names,
    )
    if train_result is None:
        print("No training data could be generated.")
        return PreparedData(None, None, None, None)
    input_dim = train_result.X.shape[1]

    preprocess_doxascope_data(
        val_files,
        preprocessed_dir,
        "val.npz",
        num_future_timesteps,
        num_past_timesteps,
        granularity=granularity,
        include_inventory=include_inventory,
        resource_names_override=resource_names,
    )

    preprocess_doxascope_data(
        test_files,
        preprocessed_dir,
        "test.npz",
        num_future_timesteps,
        num_past_timesteps,
        granularity=granularity,
        include_inventory=include_inventory,
        resource_names_override=resource_names,
    )

    # Create datasets and dataloaders from loaded/processed data
    train_loader, val_loader, test_loader = None, None, None

    train_data = np.load(preprocessed_dir / "train.npz", allow_pickle=True)
    y_inventory = train_data["y_inventory"] if "y_inventory" in train_data else None
    time_to_change = train_data["time_to_change"] if "time_to_change" in train_data else None
    train_dataset = DoxascopeDataset(train_data["X"], train_data["y"], y_inventory, time_to_change)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"  Train samples: {len(train_dataset)}")

    if (preprocessed_dir / "val.npz").exists():
        val_data = np.load(preprocessed_dir / "val.npz", allow_pickle=True)
        y_inv_val = val_data["y_inventory"] if "y_inventory" in val_data else None
        ttc_val = val_data["time_to_change"] if "time_to_change" in val_data else None
        val_dataset = DoxascopeDataset(val_data["X"], val_data["y"], y_inv_val, ttc_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"  Validation samples: {len(val_dataset)}")

    if (preprocessed_dir / "test.npz").exists():
        test_data = np.load(preprocessed_dir / "test.npz", allow_pickle=True)
        y_inv_test = test_data["y_inventory"] if "y_inventory" in test_data else None
        ttc_test = test_data["time_to_change"] if "time_to_change" in test_data else None
        test_dataset = DoxascopeDataset(test_data["X"], test_data["y"], y_inv_test, ttc_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"  Test samples: {len(test_dataset)}")

    inventory_num_items = len(resource_names) if resource_names else None

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        resource_names=resource_names,
        inventory_num_items=inventory_num_items,
    )
