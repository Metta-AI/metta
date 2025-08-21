"""
Generic torch.nn.Module serialization for architecture comparison.

Extracts and saves the structure, parameters, and configuration of a PyTorch model
without the forward method logic, allowing cross-version comparison.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ModelArchitectureSerializer:
    """Serializes PyTorch model architecture and parameters for comparison."""

    def __init__(self):
        self.supported_modules = {
            # Common layer types - extend as needed
            nn.Linear: self._serialize_linear,
            nn.LSTM: self._serialize_lstm,
            nn.GRU: self._serialize_gru,
            nn.RNN: self._serialize_rnn,
            nn.Conv1d: self._serialize_conv1d,
            nn.Conv2d: self._serialize_conv2d,
            nn.BatchNorm1d: self._serialize_batchnorm1d,
            nn.BatchNorm2d: self._serialize_batchnorm2d,
            nn.LayerNorm: self._serialize_layernorm,
            nn.Dropout: self._serialize_dropout,
            nn.Embedding: self._serialize_embedding,
        }

    def serialize_model(self, model: nn.Module, path: str) -> None:
        """Serialize a model's architecture and parameters.

        Args:
            model: PyTorch model to serialize
            path: File path to save serialized model
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Extract model information
        model_data = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "architecture": self._extract_architecture(model),
            "state_dict": self._serialize_state_dict(model.state_dict()),
            "named_modules": self._extract_named_modules(model),
            "model_str": str(model),
            "custom_attributes": self._extract_custom_attributes(model),
        }

        with open(path_obj, "wb") as f:
            pickle.dump(model_data, f)

    def load_generic_model(self, path: str) -> "GenericModel":
        """Load a serialized model as a generic representation.

        Args:
            path: File path to load from

        Returns:
            GenericModel instance with the loaded architecture
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Serialized model not found: {path}")

        with open(path_obj, "rb") as f:
            model_data = pickle.load(f)

        return GenericModel(model_data)

    def _extract_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Extract the module hierarchy and structure."""
        architecture = {}

        for name, module in model.named_modules():
            if name == "":  # Skip the root module
                continue

            module_info = {
                "type": module.__class__.__name__,
                "module_path": module.__class__.__module__,
            }

            # Add module-specific configuration
            if type(module) in self.supported_modules:
                module_info.update(self.supported_modules[type(module)](module))
            else:
                # For unsupported modules, save basic info
                module_info["parameters"] = {
                    name: param.shape for name, param in module.named_parameters(recurse=False)
                }
                module_info["buffers"] = {name: buf.shape for name, buf in module.named_buffers(recurse=False)}

            architecture[name] = module_info

        return architecture

    def _extract_named_modules(self, model: nn.Module) -> Dict[str, str]:
        """Extract named modules as string representations."""
        return {name: str(module) for name, module in model.named_modules()}

    def _extract_custom_attributes(self, model: nn.Module) -> Dict[str, Any]:
        """Extract custom attributes that might affect model behavior."""
        custom_attrs = {}

        # Get all attributes that aren't standard nn.Module attributes
        standard_attrs = set(dir(nn.Module()))
        model_attrs = set(dir(model))
        custom_attr_names = model_attrs - standard_attrs

        for attr_name in custom_attr_names:
            if not attr_name.startswith("_"):  # Skip private attributes
                try:
                    attr_value = getattr(model, attr_name)
                    # Only save serializable attributes
                    if isinstance(attr_value, (int, float, str, bool, list, tuple, dict)):
                        custom_attrs[attr_name] = attr_value
                except:
                    pass  # Skip attributes that can't be accessed

        return custom_attrs

    def _serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Serialize state dict with shape and dtype info."""
        serialized = {}
        for key, tensor in state_dict.items():
            serialized[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "requires_grad": tensor.requires_grad,
                "device": str(tensor.device),
                "data": tensor.detach().cpu().numpy().tobytes(),  # Actual tensor data
            }
        return serialized

    # Module-specific serialization methods
    def _serialize_linear(self, module: nn.Linear) -> Dict[str, Any]:
        return {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }

    def _serialize_lstm(self, module: nn.LSTM) -> Dict[str, Any]:
        return {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers,
            "bias": module.bias,
            "batch_first": module.batch_first,
            "dropout": module.dropout,
            "bidirectional": module.bidirectional,
        }

    def _serialize_gru(self, module: nn.GRU) -> Dict[str, Any]:
        return {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers,
            "bias": module.bias,
            "batch_first": module.batch_first,
            "dropout": module.dropout,
            "bidirectional": module.bidirectional,
        }

    def _serialize_rnn(self, module: nn.RNN) -> Dict[str, Any]:
        return {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers,
            "nonlinearity": module.nonlinearity,
            "bias": module.bias,
            "batch_first": module.batch_first,
            "dropout": module.dropout,
            "bidirectional": module.bidirectional,
        }

    def _serialize_conv1d(self, module: nn.Conv1d) -> Dict[str, Any]:
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
            "bias": module.bias is not None,
        }

    def _serialize_conv2d(self, module: nn.Conv2d) -> Dict[str, Any]:
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
            "bias": module.bias is not None,
        }

    def _serialize_batchnorm1d(self, module: nn.BatchNorm1d) -> Dict[str, Any]:
        return {
            "num_features": module.num_features,
            "eps": module.eps,
            "momentum": module.momentum,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
        }

    def _serialize_batchnorm2d(self, module: nn.BatchNorm2d) -> Dict[str, Any]:
        return {
            "num_features": module.num_features,
            "eps": module.eps,
            "momentum": module.momentum,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
        }

    def _serialize_layernorm(self, module: nn.LayerNorm) -> Dict[str, Any]:
        return {
            "normalized_shape": list(module.normalized_shape),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
        }

    def _serialize_dropout(self, module: nn.Dropout) -> Dict[str, Any]:
        return {
            "p": module.p,
            "inplace": module.inplace,
        }

    def _serialize_embedding(self, module: nn.Embedding) -> Dict[str, Any]:
        return {
            "num_embeddings": module.num_embeddings,
            "embedding_dim": module.embedding_dim,
            "padding_idx": module.padding_idx,
            "max_norm": module.max_norm,
            "norm_type": module.norm_type,
            "scale_grad_by_freq": module.scale_grad_by_freq,
            "sparse": module.sparse,
        }


class GenericModel:
    """Generic representation of a PyTorch model without forward logic."""

    def __init__(self, model_data: Dict[str, Any]):
        self.model_class = model_data["model_class"]
        self.model_module = model_data["model_module"]
        self.architecture = model_data["architecture"]
        self.state_dict_info = model_data["state_dict"]
        self.named_modules = model_data["named_modules"]
        self.model_str = model_data["model_str"]
        self.custom_attributes = model_data["custom_attributes"]

    def compare_architecture(self, other: "GenericModel") -> Tuple[bool, str]:
        """Compare this model's architecture with another."""

        # Compare basic info
        if self.model_class != other.model_class:
            return False, f"Different model classes: {self.model_class} vs {other.model_class}"

        # Compare module structure
        if set(self.architecture.keys()) != set(other.architecture.keys()):
            return False, "Different module structure"

        # Compare each module
        for name in self.architecture.keys():
            self_module = self.architecture[name]
            other_module = other.architecture[name]

            if self_module["type"] != other_module["type"]:
                return False, f"Module {name}: different types {self_module['type']} vs {other_module['type']}"

            # Compare module-specific configuration
            for key in self_module.keys():
                if key in ["type", "module_path"]:
                    continue
                if self_module.get(key) != other_module.get(key):
                    return False, f"Module {name}: {key} differs"

        # Compare state dict structure
        if set(self.state_dict_info.keys()) != set(other.state_dict_info.keys()):
            return False, "Different parameter names"

        for param_name in self.state_dict_info.keys():
            self_param = self.state_dict_info[param_name]
            other_param = other.state_dict_info[param_name]

            if self_param["shape"] != other_param["shape"]:
                return False, f"Parameter {param_name}: different shapes"

            if self_param["dtype"] != other_param["dtype"]:
                return False, f"Parameter {param_name}: different dtypes"

        return True, "Architectures match"

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Reconstruct the state dict from serialized data."""
        state_dict = {}
        for key, param_info in self.state_dict_info.items():
            # Reconstruct tensor from bytes
            import numpy as np

            dtype = getattr(torch, param_info["dtype"].split(".")[-1])
            data = np.frombuffer(param_info["data"], dtype=np.uint8)
            # Note: This assumes the original dtype mapping, you might need to adjust
            tensor = torch.from_numpy(data).view(param_info["shape"])
            state_dict[key] = tensor
        return state_dict

    def summary(self) -> str:
        """Get a summary of the model architecture."""
        summary = f"Model: {self.model_class}\n"
        summary += f"Modules: {len(self.architecture)}\n"
        summary += f"Parameters: {len(self.state_dict_info)}\n"
        summary += f"Custom attributes: {list(self.custom_attributes.keys())}\n"
        return summary


# Convenience functions
def save_model_architecture(model: nn.Module, path: str) -> None:
    """Save a model's architecture for later comparison."""
    serializer = ModelArchitectureSerializer()
    serializer.serialize_model(model, path)


def load_model_architecture(path: str) -> GenericModel:
    """Load a saved model architecture."""
    serializer = ModelArchitectureSerializer()
    return serializer.load_generic_model(path)


def compare_model_architectures(path1: str, path2: str) -> Tuple[bool, str]:
    """Compare two saved model architectures."""
    model1 = load_model_architecture(path1)
    model2 = load_model_architecture(path2)
    return model1.compare_architecture(model2)
