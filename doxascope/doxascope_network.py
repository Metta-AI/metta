#!/usr/bin/env python3
"""
Doxascope Neural Network

A PyTorch implementation of a neural network that predicts agent movement from LSTM memory vectors.
"""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
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
        shared_head_dim=0,
        skip_connection_weight=0.1,
    ):
        super(DoxascopeNet, self).__init__()

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "num_future_timesteps": num_future_timesteps,
            "activation_fn": activation_fn,
            "main_net_depth": main_net_depth,
            "processor_depth": processor_depth,
            "shared_head_dim": shared_head_dim,
            "skip_connection_weight": skip_connection_weight,
        }

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_future_timesteps = num_future_timesteps
        self.skip_connection_weight = skip_connection_weight
        act_fn = get_activation_fn(activation_fn)

        # Assume input is [hidden_state, cell_state] concatenated
        lstm_state_dim = input_dim // 2
        processor_output_dim = hidden_dim // 2

        # Build processor networks dynamically
        def build_processor(depth):
            layers = []
            in_dim = lstm_state_dim
            for i in range(depth):
                out_dim = processor_output_dim
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout_rate))
                in_dim = out_dim
            return nn.Sequential(*layers)

        self.hidden_processor = build_processor(processor_depth)
        self.cell_processor = build_processor(processor_depth)

        # Build main processing network dynamically
        main_net_layers = []
        main_in_dim = hidden_dim
        main_out_dim = hidden_dim // 2

        if main_net_depth > 0:
            if main_net_depth == 1:
                main_net_layers.extend([nn.Linear(main_in_dim, main_out_dim), act_fn, nn.Dropout(dropout_rate)])
            else:
                # Input layer
                main_net_layers.extend([nn.Linear(main_in_dim, hidden_dim), act_fn, nn.Dropout(dropout_rate)])
                # Hidden layers
                for _ in range(main_net_depth - 2):
                    main_net_layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Dropout(dropout_rate)])
                # Output layer
                main_net_layers.extend([nn.Linear(hidden_dim, main_out_dim), act_fn, nn.Dropout(dropout_rate)])

        self.main_net = nn.Sequential(*main_net_layers)

        # Optional shared head layer
        self.shared_head_net = None
        head_input_dim = main_out_dim
        if shared_head_dim > 0:
            self.shared_head_net = nn.Sequential(
                nn.Linear(head_input_dim, shared_head_dim), act_fn, nn.Dropout(dropout_rate)
            )
            head_input_dim = shared_head_dim

        # Create k separate output heads
        self.output_heads = nn.ModuleList([nn.Linear(head_input_dim, num_classes) for _ in range(num_future_timesteps)])

        # Skip connection (kept for stability, minimal impact)
        self.skip_connection = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_state_dim = self.input_dim // 2

        # Split into hidden and cell states
        hidden_state = x[:, :lstm_state_dim]
        cell_state = x[:, lstm_state_dim:]

        # Process separately (OPTIMIZED: separate processing is crucial)
        h_processed = self.hidden_processor(hidden_state)
        c_processed = self.cell_processor(cell_state)

        # Combine processed features
        combined = torch.cat([h_processed, c_processed], dim=1)

        # Main prediction
        main_features = self.main_net(combined)

        # Optional shared head
        if self.shared_head_net:
            main_features = self.shared_head_net(main_features)

        # Generate output from each head
        outputs = [head(main_features) for head in self.output_heads]

        # Skip connection for residual learning (minimal impact but kept for stability)
        # We apply the skip connection to the first timestep's prediction
        if self.skip_connection_weight > 0:
            skip_output = self.skip_connection(x)
            outputs[0] = outputs[0] + self.skip_connection_weight * skip_output  # Weighted skip connection

        return outputs


class DoxascopeTrainer:
    """Training and evaluation pipeline for the doxascope network."""

    def __init__(self, model, output_dir: Path, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_accuracies_per_step = []

    def _compute_loss(self, outputs, batch_y, criterion):
        """Computes the total loss across all prediction heads."""
        loss = torch.tensor(0.0, device=self.device)
        for i, out in enumerate(outputs):
            target = batch_y[:, i]
            loss += criterion(out, target)
        return loss

    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total = 0

        num_steps = self.model.num_future_timesteps
        correct_per_step = [0] * num_steps

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)

            # Sum losses from all heads
            loss = self._compute_loss(outputs, batch_y, criterion)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += batch_y.size(0)

            # Accuracy per step
            for i in range(num_steps):
                _, predicted = outputs[i].max(1)
                correct_per_step[i] += predicted.eq(batch_y[:, i]).sum().item()

        acc_per_step = [100.0 * c / total for c in correct_per_step] if total > 0 else [0.0] * num_steps
        return total_loss / len(dataloader), acc_per_step

    def evaluate(self, dataloader, criterion):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total = 0

        # For multi-step, track accuracy and predictions per step
        num_steps = self.model.num_future_timesteps
        correct_per_step = [0] * num_steps
        all_preds_per_step = [[] for _ in range(num_steps)]
        all_targets_per_step = [[] for _ in range(num_steps)]

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x)

                # Sum losses from all heads
                loss = self._compute_loss(outputs, batch_y, criterion)

                total_loss += loss.item()

                # Collect accuracy, predictions, and targets for each step
                for i in range(num_steps):
                    _, predicted = outputs[i].max(1)
                    correct_per_step[i] += predicted.eq(batch_y[:, i]).sum().item()
                    all_preds_per_step[i].extend(predicted.cpu().numpy())
                    all_targets_per_step[i].extend(batch_y[:, i].cpu().numpy())

                total += batch_y.size(0)

        # Calculate accuracies
        acc_per_step = [100.0 * c / total for c in correct_per_step] if total > 0 else [0.0] * num_steps
        # Overall accuracy is the average across all timesteps
        acc = sum(acc_per_step) / len(acc_per_step) if acc_per_step else 0.0

        if len(dataloader) == 0:
            return 0.0, acc, acc_per_step, all_preds_per_step, all_targets_per_step

        return total_loss / len(dataloader), acc, acc_per_step, all_preds_per_step, all_targets_per_step

    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10):
        """Train the model with early stopping."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience_counter = 0

        # Save initial model state
        initial_model_state = {
            "config": self.model.config,
            "state_dict": self.model.state_dict(),
        }
        torch.save(initial_model_state, self.output_dir / "best_model.pth")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc_per_step = self.train_epoch(train_loader, optimizer, criterion)

            # Evaluate
            val_loss, val_acc, val_acc_per_step, _, _ = self.evaluate(val_loader, criterion)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc_per_step[0])
            self.val_accuracies.append(val_acc)
            self.val_accuracies_per_step.append(val_acc_per_step)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save the best model state and config
                best_model_state = {
                    "config": self.model.config,
                    "state_dict": self.model.state_dict(),
                }
                torch.save(best_model_state, self.output_dir / "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print progress
            epoch_time = time.time() - start_time
            train_acc_str = ", ".join([f"{acc:.1f}%" for acc in train_acc_per_step])
            val_acc_str = ", ".join([f"{acc:.1f}%" for acc in val_acc_per_step])

            print(
                f"Epoch {epoch + 1:3d}: "
                f"Train Loss: {train_loss:.4f}, Accs: [{train_acc_str}] | "
                f"Val Loss: {val_loss:.4f}, Accs: [{val_acc_str}] | "
                f"Time: {epoch_time:.2f}s"
            )

        # Restore the best model
        best_model_checkpoint = torch.load(self.output_dir / "best_model.pth")
        self.model.load_state_dict(best_model_checkpoint["state_dict"])
        print(f"🎯 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc

    def analyze_results(self, test_loader, timesteps_to_analyze=(1,)):
        """Analyze and visualize results."""
        # Get predictions
        _, test_acc, test_acc_per_step, all_preds_per_step, all_targets_per_step = self.evaluate(
            test_loader, nn.CrossEntropyLoss()
        )
        movement_names = ["Stay", "Up", "Down", "Left", "Right"]

        # Generate plots
        self.plot_training_curves()

        # Generate confusion matrix for specified timesteps
        for t in timesteps_to_analyze:
            # Ensure the requested timestep is valid
            if 1 <= t <= len(all_preds_per_step):
                preds = all_preds_per_step[t - 1]
                targets = all_targets_per_step[t - 1]
                self.plot_confusion_matrix(targets, preds, movement_names, timestep=t)

        self.plot_multistep_accuracy(test_acc_per_step)

        print(f"Final Test Accuracy (t+1): {test_acc:.2f}%")

        return test_acc

    def plot_training_curves(self):
        """Plot training and validation loss/accuracy curves."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label="Train Acc (t+1)")
        plt.plot(self.val_accuracies, label="Val Acc (t+1)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()

    def plot_confusion_matrix(self, targets, preds, movement_names, timestep: int):
        """Plot the confusion matrix."""
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=movement_names,
            yticklabels=movement_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (t+{timestep})")
        plt.savefig(self.output_dir / f"confusion_matrix_t+{timestep}.png")
        plt.close()

    def plot_multistep_accuracy(self, test_acc_per_step):
        """Plots test accuracy for each future timestep."""
        plt.figure(figsize=(10, 6))
        steps = range(1, len(test_acc_per_step) + 1)
        plt.plot(steps, test_acc_per_step, marker="o", linestyle="-")
        plt.title("Test Accuracy per Future Timestep")
        plt.xlabel("Future Timestep (t+k)")
        plt.ylabel("Test Accuracy (%)")
        plt.grid(True)
        plt.xticks(steps)
        plt.tight_layout()
        plt.savefig(self.output_dir / "multistep_accuracy.png")
        plt.close()


def prepare_data(
    raw_data_dir: Path,
    output_dir: Path,
    test_split: float,
    val_split: float,
    num_future_timesteps: int,
):
    """Prepares training, validation, and test datasets."""
    all_json_files = sorted(list(raw_data_dir.glob("*.json")))
    if not all_json_files:
        raise FileNotFoundError(f"No raw data files found in {raw_data_dir}")

    random.shuffle(all_json_files)
    num_files = len(all_json_files)
    test_idx = int(num_files * test_split)
    val_idx = int(num_files * (test_split + val_split))

    test_files = all_json_files[:test_idx]
    val_files = all_json_files[test_idx:val_idx]
    train_files = all_json_files[val_idx:]

    print(f"Found {num_files} total files: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    preprocessed_dir = output_dir / "preprocessed_data"
    X_train, y_train = preprocess_doxascope_data(train_files, preprocessed_dir, "train_data.npz", num_future_timesteps)
    X_val, y_val = preprocess_doxascope_data(val_files, preprocessed_dir, "val_data.npz", num_future_timesteps)
    X_test, y_test = preprocess_doxascope_data(test_files, preprocessed_dir, "test_data.npz", num_future_timesteps)

    if X_train is None or X_val is None or X_test is None:
        raise ValueError("Data preprocessing failed. Check logs for details.")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_doxascope(
    raw_data_dir: Path,
    output_dir: Path,
    batch_size=32,
    test_split=0.2,
    val_split=0.1,
    num_epochs=100,
    lr=0.0007,
    num_future_timesteps=1,
):
    """
    Main function to train the doxascope network.
    """
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(
            raw_data_dir, output_dir, test_split, val_split, num_future_timesteps
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Create datasets and dataloaders
    train_dataset = DoxascopeDataset(X_train, y_train)
    val_dataset = DoxascopeDataset(X_val, y_val)
    test_dataset = DoxascopeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    input_dim = X_train.shape[1]
    model_params = {
        "input_dim": input_dim,
        "hidden_dim": 512,
        "num_classes": 5,
        "dropout_rate": 0.4,
        "num_future_timesteps": num_future_timesteps,
        "activation_fn": "silu",
        "main_net_depth": 3,
        "processor_depth": 1,
        "shared_head_dim": 256,
    }
    model = DoxascopeNet(**model_params)

    # Training
    trainer = DoxascopeTrainer(model, output_dir, device)
    best_val_acc = trainer.train(train_loader, val_loader, num_epochs=num_epochs, lr=lr)

    # Analysis
    print("\n--- Running Final Analysis ---")
    trainer.plot_training_curves()

    # Analyze specific timesteps for confusion matrix
    timesteps_to_analyze = [1]
    if num_future_timesteps >= 20:
        timesteps_to_analyze.append(20)

    (
        test_loss,
        test_acc,
        test_acc_per_step,
        all_preds_per_step,
        all_targets_per_step,
    ) = trainer.evaluate(test_loader, nn.CrossEntropyLoss())

    trainer.analyze_results(
        test_loader,
        timesteps_to_analyze=timesteps_to_analyze,
    )
    trainer.plot_multistep_accuracy(test_acc_per_step)

    end_time = time.time()
    print(f"\n✅ Training and analysis complete in {end_time - start_time:.2f} seconds.")
    print(f"📈 Final Test Accuracy: {test_acc:.2f}%")
    print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}%")

    # Return for potential further use, e.g., in automated scripts
    return best_val_acc, test_acc
