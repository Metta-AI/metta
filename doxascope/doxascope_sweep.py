#!/usr/bin/env python3
"""
Doxascope Sweeps

Unified script for running various hyperparameter and architectural sweeps
for the doxascope network.
"""

import itertools
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .doxascope_network import DoxascopeDataset, DoxascopeNet, DoxascopeTrainer

# ======================================================================================
#
# ARCHITECTURAL SWEEP
#
# ======================================================================================


class ArchitecturalVariant(nn.Module):
    """Doxascope network variant for testing architectural choices."""

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_classes=5,
        dropout_rate=0.3,
        use_attention=True,
        use_skip_connection=True,
        separate_processing=True,
        activation="relu",
    ):
        super(ArchitecturalVariant, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_skip_connection = use_skip_connection
        self.separate_processing = separate_processing

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

        lstm_state_dim = input_dim // 2

        if separate_processing:
            # Separate processing pathways
            self.hidden_processor = nn.Sequential(
                nn.Linear(lstm_state_dim, hidden_dim // 2), self.activation, nn.Dropout(dropout_rate)
            )
            self.cell_processor = nn.Sequential(
                nn.Linear(lstm_state_dim, hidden_dim // 2), self.activation, nn.Dropout(dropout_rate)
            )
            combined_dim = hidden_dim
        else:
            # Unified processing
            self.unified_processor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), self.activation, nn.Dropout(dropout_rate)
            )
            combined_dim = hidden_dim

        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                self.activation,
                nn.Linear(combined_dim // 2, combined_dim),
                nn.Sigmoid(),
            )

        # Main processing network
        self.main_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Skip connection (optional)
        if use_skip_connection:
            self.skip_connection = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if self.separate_processing:
            # Split and process separately
            lstm_state_dim = self.input_dim // 2
            hidden_state = x[:, :lstm_state_dim]
            cell_state = x[:, lstm_state_dim:]

            h_processed = self.hidden_processor(hidden_state)
            c_processed = self.cell_processor(cell_state)
            combined = torch.cat([h_processed, c_processed], dim=1)
        else:
            # Unified processing
            combined = self.unified_processor(x)

        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(combined)
            attended = combined * attention_weights
        else:
            attended = combined
            attention_weights = torch.ones_like(combined)

        # Main prediction
        main_output = self.main_net(attended)

        # Skip connection if enabled
        if self.use_skip_connection:
            skip_output = self.skip_connection(x)
            output = main_output + 0.1 * skip_output
        else:
            output = main_output

        return output, attention_weights


def test_architectural_variant(config, X, y, max_epochs=50):
    """Test a single architectural configuration."""
    try:
        # Create dataset splits
        dataset = DoxascopeDataset(X, y)
        total_size = len(dataset)
        test_size = int(total_size * 0.2)
        val_size = int(total_size * 0.1)
        train_size = total_size - test_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        batch_size = config.get("batch_size", 64)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ArchitecturalVariant(
            input_dim=X.shape[1],
            hidden_dim=config.get("hidden_dim", 256),
            num_classes=len(np.unique(y)),
            dropout_rate=config.get("dropout_rate", 0.3),
            use_attention=config.get("use_attention", True),
            use_skip_connection=config.get("use_skip_connection", True),
            separate_processing=config.get("separate_processing", True),
            activation=config.get("activation", "relu"),
        )

        trainer = DoxascopeTrainer(model, output_dir=Path("./temp_sweep_models/"))

        # Training
        start_time = time.time()
        val_acc = trainer.train(train_loader, val_loader, num_epochs=max_epochs, lr=config.get("lr", 0.001))
        train_time = time.time() - start_time

        # Test evaluation
        _, test_acc, _, _ = trainer.evaluate(test_loader, nn.CrossEntropyLoss())

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        return {
            "config": config,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "train_time": train_time,
            "param_count": param_count,
            "success": True,
        }

    except Exception as e:
        return {"config": config, "error": str(e), "success": False}


def architectural_sweep(policy_name: str):
    """Run architectural sweep."""
    print("🏗️ Doxascope Architectural Sweep")
    print("=" * 50)

    # Load data
    data_path = f"../data/results/{policy_name}/training_data.npz"
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found for policy '{policy_name}' at {data_path}")
        print("👉 Please run the training script first to generate the data: ")
        print(f"   python -m doxascope.doxascope_train {policy_name}")
        return

    X, y = data["X"], data["y"]
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Define architectural variants to test
    base_config = {
        "hidden_dim": 256,
        "dropout_rate": 0.3,
        "lr": 0.001,
        "batch_size": 64,
    }

    # Architectural choices to test
    architectural_configs = [
        # Baseline
        {
            **base_config,
            "use_attention": True,
            "use_skip_connection": True,
            "separate_processing": True,
            "activation": "relu",
            "name": "Baseline (Full Architecture)",
        },
        # Test attention
        {
            **base_config,
            "use_attention": False,
            "use_skip_connection": True,
            "separate_processing": True,
            "activation": "relu",
            "name": "No Attention",
        },
        # Test skip connection
        {
            **base_config,
            "use_attention": True,
            "use_skip_connection": False,
            "separate_processing": True,
            "activation": "relu",
            "name": "No Skip Connection",
        },
        # Test separate processing
        {
            **base_config,
            "use_attention": True,
            "use_skip_connection": True,
            "separate_processing": False,
            "activation": "relu",
            "name": "Unified Processing",
        },
        # Test activation functions
        {
            **base_config,
            "use_attention": True,
            "use_skip_connection": True,
            "separate_processing": True,
            "activation": "gelu",
            "name": "GELU Activation",
        },
        {
            **base_config,
            "use_attention": True,
            "use_skip_connection": True,
            "separate_processing": True,
            "activation": "swish",
            "name": "Swish Activation",
        },
        # Minimal architecture
        {
            **base_config,
            "use_attention": False,
            "use_skip_connection": False,
            "separate_processing": False,
            "activation": "relu",
            "name": "Minimal Architecture",
        },
        # Optimized configuration (no attention, GELU)
        {
            **base_config,
            "use_attention": False,
            "use_skip_connection": True,
            "separate_processing": True,
            "activation": "gelu",
            "name": "Optimized (No Attention + GELU)",
        },
    ]

    results = []

    print(f"🔍 Testing {len(architectural_configs)} architectural variants")

    for i, config in enumerate(architectural_configs):
        print(f"\n⚡ Variant {i + 1}/{len(architectural_configs)}: {config['name']}")
        print(f"   Config: {config}")

        result = test_architectural_variant(config, X, y)
        results.append(result)

        if result["success"]:
            test_acc = result["test_accuracy"]
            val_acc = result["val_accuracy"]
            params = result["param_count"]
            time_taken = result["train_time"]

            print(f"   ✅ Test: {test_acc:.2f}%, Val: {val_acc:.2f}%")
            print(f"   📊 Params: {params:,}, Time: {time_taken:.1f}s")

        else:
            print(f"   ❌ Failed: {result['error']}")

    # Save results
    output_dir = Path(f"../data/results/{policy_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "architectural_sweep_results.json"
    with open(output_file, "w") as f:
        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            if result["success"]:
                json_results.append(
                    {
                        "name": result["config"]["name"],
                        "config": {k: v for k, v in result["config"].items() if k != "name"},
                        "val_accuracy": float(result["val_accuracy"]),
                        "test_accuracy": float(result["test_accuracy"]),
                        "train_time": float(result["train_time"]),
                        "param_count": int(result["param_count"]),
                    }
                )
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    print("\n✅ Architectural sweep complete!")
    print(f"👉 To analyze, run: python -m doxascope.doxascope_analysis analyze-sweep {output_file}")
    return results


# ======================================================================================
#
# PARAMETER SWEEP
#
# ======================================================================================


class DoxascopeNetParametric(nn.Module):
    """Parametric version of DoxascopeNet for hyperparameter sweeps."""

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_classes=5,
        dropout_rate=0.3,
        num_layers=2,
        use_attention=True,
        use_skip_connection=True,
        skip_weight=0.1,
        separate_processing=True,
        activation="relu",
    ):
        super(DoxascopeNetParametric, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_skip_connection = use_skip_connection
        self.skip_weight = skip_weight
        self.separate_processing = separate_processing

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

        lstm_state_dim = input_dim // 2

        if separate_processing:
            # Separate processing pathways
            self.hidden_processor = self._make_processor(lstm_state_dim, hidden_dim // 2, num_layers, dropout_rate)
            self.cell_processor = self._make_processor(lstm_state_dim, hidden_dim // 2, num_layers, dropout_rate)
            combined_dim = hidden_dim
        else:
            # Single pathway processing
            self.unified_processor = self._make_processor(input_dim, hidden_dim, num_layers, dropout_rate)
            combined_dim = hidden_dim

        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                self.activation,
                nn.Linear(combined_dim // 2, combined_dim),
                nn.Sigmoid(),
            )

        # Main processing network
        self.main_net = self._make_classifier(combined_dim, hidden_dim, num_classes, num_layers, dropout_rate)

        # Skip connection (optional)
        if use_skip_connection:
            self.skip_connection = nn.Linear(input_dim, num_classes)

    def _make_processor(self, input_dim, output_dim, num_layers, dropout_rate):
        """Create a processor network with variable depth."""
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer
                layers.extend([nn.Linear(current_dim, output_dim), self.activation, nn.Dropout(dropout_rate)])
            else:
                # Intermediate layers
                next_dim = max(output_dim, current_dim // 2)
                layers.extend([nn.Linear(current_dim, next_dim), self.activation, nn.Dropout(dropout_rate)])
                current_dim = next_dim

        return nn.Sequential(*layers)

    def _make_classifier(self, input_dim, hidden_dim, num_classes, num_layers, dropout_rate):
        """Create classifier network with variable depth."""
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer
                layers.append(nn.Linear(current_dim, num_classes))
            else:
                # Intermediate layers
                next_dim = hidden_dim // (2**i) if i > 0 else hidden_dim
                next_dim = max(next_dim, num_classes * 2)  # Don't go too small
                layers.extend([nn.Linear(current_dim, next_dim), self.activation, nn.Dropout(dropout_rate)])
                current_dim = next_dim

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.separate_processing:
            # Split and process separately
            lstm_state_dim = self.input_dim // 2
            hidden_state = x[:, :lstm_state_dim]
            cell_state = x[:, lstm_state_dim:]

            h_processed = self.hidden_processor(hidden_state)
            c_processed = self.cell_processor(cell_state)
            combined = torch.cat([h_processed, c_processed], dim=1)
        else:
            # Unified processing
            combined = self.unified_processor(x)

        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(combined)
            attended = combined * attention_weights
        else:
            attended = combined
            attention_weights = torch.ones_like(combined)

        # Main prediction
        main_output = self.main_net(attended)

        # Skip connection if enabled
        if self.use_skip_connection:
            skip_output = self.skip_connection(x)
            output = main_output + self.skip_weight * skip_output
        else:
            output = main_output

        return output, attention_weights


class ParameterSweep:
    """Manages hyperparameter sweep experiments."""

    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.results = []
        self.best_result = None

        # Load data once
        print("📊 Loading training data...")
        data_path = f"../data/results/{policy_name}/training_data.npz"
        try:
            data = np.load(data_path)
        except FileNotFoundError:
            print(f"❌ Error: Data file not found for policy '{policy_name}' at {data_path}")
            print("👉 Please run the training script first to generate the data: ")
            print(f"   python -m doxascope.doxascope_train {policy_name}")
            # Exit gracefully
            raise SystemExit

        self.X, self.y = data["X"], data["y"]
        print(f"Dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def define_search_space(self):
        """Define the hyperparameter search space."""
        return {
            # Architecture parameters
            "hidden_dim": [128, 256, 512],
            "num_layers": [1, 2, 3],
            "dropout_rate": [0.1, 0.3, 0.5],
            "activation": ["relu", "gelu"],
            # Training parameters
            "lr": [0.0001, 0.001, 0.01],
            "batch_size": [32, 64, 128],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            # Architectural choices
            "use_attention": [True, False],
            "use_skip_connection": [True, False],
            "skip_weight": [0.05, 0.1, 0.2],
            "separate_processing": [True, False],
        }

    def sample_configurations(self, num_samples=50):
        """Sample random configurations from search space."""
        search_space = self.define_search_space()

        configs = []
        for _ in range(num_samples):
            config = {}
            for param, values in search_space.items():
                config[param] = np.random.choice(values)
            configs.append(config)

        return configs

    def grid_search_core_params(self):
        """Grid search over core hyperparameters."""
        core_params = {
            "hidden_dim": [128, 256, 384, 512],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "lr": [0.0001, 0.0005, 0.001, 0.005],
            "batch_size": [16, 32, 64, 128],
            "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
        }

        # Generate all combinations (limit to reasonable number)
        param_names = list(core_params.keys())
        param_values = [core_params[name] for name in param_names]

        configs = []
        for values in product(*param_values):
            config = dict(zip(param_names, values, strict=False))
            # Add fixed architectural choices for grid search
            config.update(
                {
                    "activation": "gelu",
                    "use_attention": False,
                    "use_skip_connection": True,
                    "skip_weight": 0.1,
                    "separate_processing": True,
                    "num_layers": 2,
                }
            )
            configs.append(config)

        return configs[:100]  # Limit for computational feasibility

    def train_single_config(self, config, max_epochs=50):
        """Train a single configuration and return results."""
        try:
            # Create dataset
            dataset = DoxascopeDataset(self.X, self.y)

            # Split data
            total_size = len(dataset)
            test_size = int(total_size * 0.15)
            val_size = int(total_size * 0.15)
            train_size = total_size - test_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
            )

            # Create data loaders
            batch_size = config.get("batch_size", 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Create model
            model = DoxascopeNetParametric(
                input_dim=self.X.shape[1],
                hidden_dim=config.get("hidden_dim", 256),
                num_classes=len(np.unique(self.y)),
                dropout_rate=config.get("dropout_rate", 0.3),
                num_layers=config.get("num_layers", 2),
                use_attention=config.get("use_attention", True),
                use_skip_connection=config.get("use_skip_connection", True),
                skip_weight=config.get("skip_weight", 0.1),
                separate_processing=config.get("separate_processing", True),
                activation=config.get("activation", "relu"),
            )
            config_hash = hash(json.dumps(config, sort_keys=True))
            trainer = DoxascopeTrainer(model, output_dir=Path(f"./temp_sweep_models/{config_hash}/"))

            # Train with early stopping
            start_time = time.time()

            # Override trainer parameters
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("lr", 0.001),
                weight_decay=config.get("weight_decay", 1e-4),
            )

            val_acc = trainer.train(
                train_loader, val_loader, num_epochs=max_epochs, optimizer=optimizer, criterion=criterion
            )

            # Load best model and test
            _, test_acc, _, _ = trainer.evaluate(test_loader, criterion)

            train_time = time.time() - start_time

            return {
                "config": config,
                "train_time": train_time,
                "best_val_acc": val_acc,
                "test_acc": test_acc,
                "success": True,
            }

        except Exception as e:
            return {
                "config": config,
                "error": str(e),
                "success": False,
            }

    def run_sweep(self, sweep_type="random", num_configs=20):
        """Run parameter sweep."""
        print(f"🚀 Starting {sweep_type} parameter sweep...")

        if sweep_type == "random":
            configs = self.sample_configurations(num_configs)
        elif sweep_type == "grid":
            configs = self.grid_search_core_params()
        else:
            raise ValueError(f"Unknown sweep type: {sweep_type}")

        print(f"📊 Testing {len(configs)} configurations")

        for i, config in enumerate(configs):
            print(f"\n⚡ Config {i + 1}/{len(configs)}")
            print(f"   {config}")

            result = self.train_single_config(config)
            self.results.append(result)

            if result["success"]:
                print(f"   ✅ Val: {result['best_val_acc']:.2f}%, Test: {result['test_acc']:.2f}%")
                print(f"   ⏱️  Time: {result['train_time']:.1f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")

        # Update best result
        successful_results = [r for r in self.results if r["success"]]
        if successful_results:
            self.best_result = max(successful_results, key=lambda x: x["test_acc"])

        return self.results

    def save_results(self, sweep_type):
        """Save sweep results to JSON."""
        output_dir = Path(f"../data/results/{self.policy_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"parameter_sweep_{sweep_type}.json"

        # Convert numpy types to native Python for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if key == "config" and isinstance(value, dict):
                    # Handle config dict
                    serializable_config = {}
                    for k, v in value.items():
                        if isinstance(v, np.integer):
                            serializable_config[k] = int(v)
                        elif isinstance(v, np.floating):
                            serializable_config[k] = float(v)
                        else:
                            serializable_config[k] = v
                    serializable_result[key] = serializable_config
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"💾 Results saved to {output_file}")
        print(f"👉 To analyze, run: python -m doxascope.doxascope_analysis analyze-sweep {output_file}")


def parameter_sweep(sweep_type: str, num_configs: int, policy_name: str):
    """Main function for parameter sweep."""
    sweep = ParameterSweep(policy_name)

    if sweep_type == "all":
        # Run random sweep first
        print("🎲 Random Parameter Sweep")
        print("=" * 50)
        sweep.run_sweep("random", num_configs=num_configs)
        sweep.save_results("random")

        # Reset and run grid sweep on core parameters
        sweep.results = []
        print("\n🔍 Grid Search on Core Parameters")
        print("=" * 50)
        sweep.run_sweep("grid")
        sweep.save_results("grid")
    else:
        sweep.run_sweep(sweep_type, num_configs=num_configs)
        sweep.save_results(sweep_type)

    print("\n🎉 Parameter sweep complete!")


# ======================================================================================
#
# QUICK SWEEP
#
# ======================================================================================


def quick_sweep(policy_name: str):
    """Quick parameter sweep to identify promising configurations."""

    print("🚀 Quick Doxascope Parameter Sweep")
    print("=" * 50)

    # Load data
    print("📊 Loading data...")
    data_path = f"../data/results/{policy_name}/training_data.npz"
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found for policy '{policy_name}' at {data_path}")
        print("👉 Please run the training script first to generate the data: ")
        print(f"   python -m doxascope.doxascope_train {policy_name}")
        return
    X, y = data["X"], data["y"]
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Define parameter ranges (smaller ranges for quick exploration)
    param_grid = {
        "hidden_dim": [128, 256, 512],
        "dropout_rate": [0.1, 0.2, 0.3],
        "learning_rate": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64],
    }

    print(f"🔍 Testing {len(list(itertools.product(*param_grid.values())))} configurations")

    results = []
    best_result = None

    for i, (hidden_dim, dropout_rate, lr, batch_size) in enumerate(itertools.product(*param_grid.values())):
        config = {
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "learning_rate": lr,
            "batch_size": batch_size,
        }

        print(f"\n⚡ Config {i + 1}: {config}")

        try:
            # Create dataset and splits
            dataset = DoxascopeDataset(X, y)
            total_size = len(dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.2 * total_size)
            test_size = total_size - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
            )

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Create model and trainer
            model = DoxascopeNet(
                input_dim=X.shape[1],
                hidden_dim=hidden_dim,
                num_classes=len(np.unique(y)),
                dropout_rate=dropout_rate,
            )
            trainer = DoxascopeTrainer(model, output_dir=Path(f"./temp_sweep_models/{i}/"))

            # Quick training (fewer epochs)
            start_time = time.time()
            val_acc = trainer.train(train_loader, val_loader, num_epochs=30, lr=lr)
            train_time = time.time() - start_time

            # Quick test evaluation
            _, test_acc, _, _ = trainer.evaluate(test_loader, nn.CrossEntropyLoss())

            result = {
                "config": config,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "train_time": train_time,
            }

            results.append(result)

            print(f"   ✅ Val: {val_acc:.2f}%, Test: {test_acc:.2f}%, Time: {train_time:.1f}s")

            # Track best result
            if best_result is None or test_acc > best_result["test_accuracy"]:
                best_result = result

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({"config": config, "error": str(e)})

    # Save results
    output_dir = Path(f"../data/results/{policy_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "quick_sweep_results.json"
    with open(output_file, "w") as f:
        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            if "error" not in result:
                json_results.append(
                    {
                        "config": result["config"],
                        "val_accuracy": float(result["val_accuracy"]),
                        "test_accuracy": float(result["test_accuracy"]),
                        "train_time": float(result["train_time"]),
                    }
                )
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    print("\n✅ Quick sweep complete!")
    print(f"👉 To analyze, run: python -m doxascope.doxascope_analysis analyze-sweep {output_file}")
    return results


# ======================================================================================
#
# CLI INTERFACE
#
# ======================================================================================


def print_usage():
    """Prints the usage instructions."""
    print("Usage: python -m doxascope.doxascope_sweep <command> <policy_name> [options]")
    print("\nCommands:")
    print("  arch <policy_name>          Run the architectural sweep.")
    print("  quick <policy_name>         Run a quick sweep.")
    print("  param <policy_name>         Run a hyperparameter sweep.")
    print("\nOptions for 'param' command:")
    print("  --type <type>         Type of sweep: 'random', 'grid', or 'all' (default: random).")
    print("  --num-configs <n>     Number of configs for random sweep (default: 30).")


def main():
    """Main CLI entrypoint."""
    args = sys.argv[1:]
    if len(args) < 2:
        print_usage()
        sys.exit(1)

    command = args[0]
    policy_name = args[1]

    if command == "arch":
        architectural_sweep(policy_name)
    elif command == "quick":
        quick_sweep(policy_name)
    elif command == "param":
        # Default values
        sweep_type = "random"
        num_configs = 30
        # Parse optional args
        i = 2
        while i < len(args):
            if args[i] == "--type" and i + 1 < len(args):
                sweep_type = args[i + 1]
                i += 2
            elif args[i] == "--num-configs" and i + 1 < len(args):
                num_configs = int(args[i + 1])
                i += 2
            else:
                print(f"Unknown option: {args[i]}")
                print_usage()
                sys.exit(1)
        parameter_sweep(sweep_type, num_configs, policy_name)
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
