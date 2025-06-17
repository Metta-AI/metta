#!/usr/bin/env python3
"""
Doxascope Neural Network

A PyTorch implementation of a neural network that predicts agent movement
from LSTM memory vectors, revealing whether the agent's memory encodes
spatial-temporal representations.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split


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

    def __init__(self, input_dim=512, hidden_dim=384, num_classes=5, dropout_rate=0.2):
        super(DoxascopeNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

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
            nn.Linear(hidden_dim // 2, num_classes),
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

        # REMOVED: Attention mechanism (simpler is better for this task)
        # Direct processing without attention

        # Main prediction
        main_output = self.main_net(combined)

        # Skip connection for residual learning (minimal impact but kept for stability)
        skip_output = self.skip_connection(x)

        # Combine outputs
        output = main_output + 0.1 * skip_output  # Weighted skip connection

        # Return dummy attention weights for compatibility
        attention_weights = torch.ones_like(combined)

        return output, attention_weights


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

    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs, attention = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        return total_loss / len(dataloader), 100.0 * correct / total

    def evaluate(self, dataloader, criterion):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs, attention = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_targets

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
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            # Evaluate
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
            else:
                patience_counter += 1

            # Print progress
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1:3d}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs (best val acc: {best_val_acc:.2f}%)")
                break

        # Load best model
        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pth"))
        print(f"🎯 Training completed! Best validation accuracy: {best_val_acc:.2f}%")

        return best_val_acc

    def analyze_results(self, test_loader):
        """Analyze and visualize results."""
        # Get predictions
        _, test_acc, preds, targets = self.evaluate(test_loader, nn.CrossEntropyLoss())
        movement_names = ["Stay", "Up", "Down", "Left", "Right"]

        # Generate plots
        self.plot_training_curves()
        self.plot_confusion_matrix(targets, preds, movement_names)

        print(f"Final Test Accuracy: {test_acc:.2f}%")

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
        plt.plot(self.train_accuracies, label="Train Acc")
        plt.plot(self.val_accuracies, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()

    def plot_confusion_matrix(self, targets, preds, movement_names):
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
        plt.title("Confusion Matrix")
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()


def train_doxascope(
    data_path: Path,
    output_dir: Path,
    batch_size=32,
    test_split=0.2,
    val_split=0.1,
    num_epochs=100,
    lr=0.001,
):
    """Main function to train the doxascope network."""
    # Load data
    try:
        with np.load(data_path) as data:
            X = data["X"]
            y = data["y"]
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}")
        return None, 0

    # Create dataset
    dataset = DoxascopeDataset(X, y)

    # Split dataset
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and trainer
    model = DoxascopeNet(input_dim=X.shape[1], num_classes=len(np.unique(y)))
    trainer = DoxascopeTrainer(model, output_dir=output_dir)

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, lr=lr)

    # Analyze results
    test_accuracy = trainer.analyze_results(test_loader)

    return trainer, test_accuracy
