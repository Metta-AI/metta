"""Tests for dormant neuron monitor component."""

import torch

from metta.rl.training.dormant_neuron_monitor import DormantNeuronMonitor, DormantNeuronMonitorConfig


class TestDormantNeuronMonitor:
    """Test cases for DormantNeuronMonitor."""

    def test_config_creation(self):
        """Test that config can be created with default values."""
        config = DormantNeuronMonitorConfig()
        assert config.epoch_interval == 1
        assert config.weight_threshold == 1e-6
        assert config.min_layer_size == 10
        assert config.track_by_layer is True
        assert config.track_overall is True

    def test_config_custom_values(self):
        """Test that config can be created with custom values."""
        config = DormantNeuronMonitorConfig(
            epoch_interval=5, weight_threshold=1e-4, min_layer_size=20, track_by_layer=False, track_overall=True
        )
        assert config.epoch_interval == 5
        assert config.weight_threshold == 1e-4
        assert config.min_layer_size == 20
        assert config.track_by_layer is False
        assert config.track_overall is True

    def test_monitor_initialization(self):
        """Test that monitor can be initialized."""
        config = DormantNeuronMonitorConfig(epoch_interval=1)
        monitor = DormantNeuronMonitor(config)

        assert monitor._enabled is True
        assert monitor._master_only is True
        assert monitor._config == config
        assert monitor._dormant_neuron_history == {}

    def test_monitor_disabled(self):
        """Test that monitor can be disabled."""
        config = DormantNeuronMonitorConfig(epoch_interval=0)
        monitor = DormantNeuronMonitor(config)

        assert monitor._enabled is False

    def test_count_dormant_neurons_linear_layer(self):
        """Test dormant neuron counting for linear layers."""
        config = DormantNeuronMonitorConfig(weight_threshold=0.1)
        monitor = DormantNeuronMonitor(config)

        # Create a linear layer with some dormant neurons
        # Neuron 0: all weights < 0.1 (dormant)
        # Neuron 1: some weights > 0.1 (active)
        # Neuron 2: all weights < 0.1 (dormant)
        weights = torch.tensor(
            [
                [0.05, 0.03, 0.02],  # Dormant neuron
                [0.05, 0.15, 0.03],  # Active neuron
                [0.02, 0.01, 0.04],  # Dormant neuron
            ]
        )

        dormant_count = monitor._count_dormant_neurons_in_layer(weights, "linear")
        assert dormant_count == 2  # Neurons 0 and 2 are dormant

    def test_count_dormant_neurons_conv_layer(self):
        """Test dormant neuron counting for conv layers."""
        config = DormantNeuronMonitorConfig(weight_threshold=0.1)
        monitor = DormantNeuronMonitor(config)

        # Create a conv layer with some dormant neurons
        # Channel 0: all weights < 0.1 (dormant)
        # Channel 1: some weights > 0.1 (active)
        weights = torch.tensor(
            [
                [[[0.05, 0.03], [0.02, 0.01]]],  # Dormant channel
                [[[0.05, 0.15], [0.03, 0.02]]],  # Active channel
            ]
        )

        dormant_count = monitor._count_dormant_neurons_in_layer(weights, "conv")
        assert dormant_count == 1  # Channel 0 is dormant

    def test_analyze_dormant_neurons_simple_network(self):
        """Test dormant neuron analysis on a simple network."""
        config = DormantNeuronMonitorConfig(weight_threshold=0.1, min_layer_size=1)
        monitor = DormantNeuronMonitor(config)

        # Create a simple network with known dormant neurons
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 2, bias=False)
                self.linear2 = torch.nn.Linear(2, 1, bias=False)

                # Set weights to create dormant neurons
                with torch.no_grad():
                    # linear1: first neuron dormant, second active
                    self.linear1.weight.data = torch.tensor(
                        [
                            [0.05, 0.03, 0.02],  # Dormant neuron
                            [0.05, 0.15, 0.03],  # Active neuron
                        ]
                    )
                    # linear2: neuron active
                    self.linear2.weight.data = torch.tensor([[0.15, 0.12]])

        network = SimpleNet()
        stats = monitor._analyze_dormant_neurons(network, epoch=1)

        # Check that we get expected statistics
        assert "dormant_neurons/linear1_weight/count" in stats
        assert "dormant_neurons/linear1_weight/ratio" in stats
        assert "dormant_neurons/linear2_weight/count" in stats
        assert "dormant_neurons/overall/count" in stats
        assert "dormant_neurons/overall/ratio" in stats

        # Check specific values
        assert stats["dormant_neurons/linear1_weight/count"] == 1.0  # One dormant neuron
        # Ratio is dormant neurons / total neurons (2 neurons total, 1 dormant = 0.5)
        assert stats["dormant_neurons/linear1_weight/ratio"] == 0.5  # 1 out of 2 neurons
        assert stats["dormant_neurons/linear2_weight/count"] == 0.0  # No dormant neurons

    def test_history_tracking(self):
        """Test that dormant neuron history is tracked correctly."""
        config = DormantNeuronMonitorConfig(weight_threshold=0.1, min_layer_size=1)
        monitor = DormantNeuronMonitor(config)

        # Create a simple network
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.linear.weight.data = torch.tensor([[0.05, 0.03]])  # Dormant neuron

        network = SimpleNet()

        # Analyze multiple times
        monitor._analyze_dormant_neurons(network, epoch=1)
        monitor._analyze_dormant_neurons(network, epoch=2)

        history = monitor.get_dormant_neuron_history()
        assert "linear_weight" in history
        assert len(history["linear_weight"]) == 2
        assert history["linear_weight"][0] == 1  # One dormant neuron in epoch 1
        assert history["linear_weight"][1] == 1  # One dormant neuron in epoch 2

    def test_skip_small_layers(self):
        """Test that very small layers are skipped."""
        config = DormantNeuronMonitorConfig(weight_threshold=0.1, min_layer_size=10)
        monitor = DormantNeuronMonitor(config)

        # Create a network with small layers
        class SmallNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1, bias=False)  # Only 2 parameters

        network = SmallNet()
        stats = monitor._analyze_dormant_neurons(network, epoch=1)

        # Since all layers are too small, we should get no stats at all
        assert len(stats) == 0
