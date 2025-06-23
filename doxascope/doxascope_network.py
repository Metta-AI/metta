#!/usr/bin/env python3
"""
Doxascope Neural Network

A PyTorch implementation of a neural network that predicts agent movement
from LSTM memory vectors, revealing whether the agent's memory encodes
spatial-temporal representations.
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

    def __init__(self, input_dim=512, hidden_dim=384, num_classes=5, dropout_rate=0.2, num_future_timesteps=1):
        super(DoxascopeNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_future_timesteps = num_future_timesteps

        # Assume input is [hidden_state, cell_state] concatenated
        lstm_state_dim = input_dim // 2

        # Separate processing for hidden and cell states
        self.hidden_processor = nn.Sequential(
            nn.Linear(lstm_state_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout_rate)
        )

        self.cell_processor = nn.Sequential(
            nn.Linear(lstm_state_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout_rate)
        )

        # Main processing network
        self.main_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Create k separate output heads
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden_dim // 2, num_classes) for _ in range(num_future_timesteps)]
        )

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

        # Generate output from each head
        outputs = [head(main_features) for head in self.output_heads]

        # Skip connection for residual learning (minimal impact but kept for stability)
        # We apply the skip connection to the first timestep's prediction
        skip_output = self.skip_connection(x)
        outputs[0] = outputs[0] + 0.1 * skip_output  # Weighted skip connection

        # Return dummy attention weights for compatibility
        attention_weights = torch.ones_like(combined)

        return outputs, attention_weights


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
            outputs, attention = self.model(batch_x)

            # Sum losses from all heads
            loss = torch.tensor(0.0, device=self.device)
            for i, out in enumerate(outputs):
                target = batch_y[:, i]
                loss += criterion(out, target)

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

                outputs, attention = self.model(batch_x)

                # Sum losses from all heads
                loss = torch.tensor(0.0, device=self.device)
                for i, out in enumerate(outputs):
                    target = batch_y[:, i]
                    loss += criterion(out, target)

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
        # Overall accuracy is based on the first timestep (t+1)
        acc = acc_per_step[0] if acc_per_step else 0.0

        return total_loss / len(dataloader), acc, acc_per_step, all_preds_per_step, all_targets_per_step

    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, optimizer=None, criterion=None):
        """Full training loop."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_acc = 0
        patience_counter = 0
        patience = 20

        print("Starting doxascope training...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc_per_step = self.train_epoch(train_loader, optimizer, criterion)

            # Evaluate
            val_loss, val_acc, val_acc_per_step, _, _ = self.evaluate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc_per_step[0])
            self.val_accuracies.append(val_acc)
            self.val_accuracies_per_step.append(val_acc_per_step)

            # Early stopping based on the primary (t+1) validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
            else:
                patience_counter += 1

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

            if patience_counter > patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

        # Load best model
        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pth"))
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
        """Plot the prediction accuracy for each future timestep."""
        if len(test_acc_per_step) <= 1:
            return

        plt.figure(figsize=(8, 5))
        steps = range(1, len(test_acc_per_step) + 1)
        plt.plot(steps, test_acc_per_step, marker="o", linestyle="-")
        plt.title("Prediction Accuracy vs. Future Timestep")
        plt.xlabel("Future Timestep (t+k)")
        plt.ylabel("Test Accuracy (%)")
        plt.xticks(steps)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "multistep_accuracy.png")
        plt.close()


def train_doxascope(
    raw_data_dir: Path,
    output_dir: Path,
    batch_size=32,
    test_split=0.2,
    val_split=0.1,
    num_epochs=100,
    lr=0.001,
    num_future_timesteps=1,
):
    """Main function to train the doxascope network."""
    # --- 1. Segregate data by file to prevent leakage ---
    print("Segregating simulation files for training, validation, and testing...")
    all_files = list(raw_data_dir.glob("doxascope_data_*.json"))
    if not all_files:
        print(f"❌ Error: No raw data files found in {raw_data_dir}")
        return None, 0

    random.shuffle(all_files)

    test_split_idx = int(len(all_files) * test_split)
    val_split_idx = test_split_idx + int(len(all_files) * val_split)

    test_files = all_files[:test_split_idx]
    val_files = all_files[test_split_idx:val_split_idx]
    train_files = all_files[val_split_idx:]

    print(f"Found {len(all_files)} files: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    # --- 2. Preprocess each dataset separately ---
    print("Preprocessing Doxascope Data...")
    X_train, y_train = preprocess_doxascope_data(
        train_files, output_dir, "train_data.npz", num_future_timesteps=num_future_timesteps
    )
    X_val, y_val = preprocess_doxascope_data(
        val_files, output_dir, "val_data.npz", num_future_timesteps=num_future_timesteps
    )
    X_test, y_test = preprocess_doxascope_data(
        test_files, output_dir, "test_data.npz", num_future_timesteps=num_future_timesteps
    )

    if X_train is None or y_train is None:
        print("❌ Error: Failed to create training data.")
        return None, 0

    # --- 3. Create datasets and dataloaders ---
    train_dataset = DoxascopeDataset(X_train, y_train)
    val_dataset = DoxascopeDataset(X_val, y_val) if X_val is not None and y_val is not None else None
    test_dataset = DoxascopeDataset(X_test, y_test) if X_test is not None and y_test is not None else None

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    if val_loader is None or test_loader is None:
        print("⚠️ Warning: Not enough data to create validation or test sets. Continuing with training only.")
        # Create dummy loaders to avoid crashing the training loop
        val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = val_loader

    # --- 4. Initialize model and trainer ---
    model = DoxascopeNet(
        input_dim=X_train.shape[1],
        num_classes=5,  # Stay, Up, Down, Left, Right
        num_future_timesteps=num_future_timesteps,
    )
    trainer = DoxascopeTrainer(model, output_dir=output_dir)

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, lr=lr)

    # Analyze results, including a distant timestep
    timesteps_to_analyze = [1]
    if num_future_timesteps >= 20:
        timesteps_to_analyze.append(20)
    test_accuracy = trainer.analyze_results(test_loader, timesteps_to_analyze=timesteps_to_analyze)

    return trainer, test_accuracy
