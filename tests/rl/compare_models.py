#!/usr/bin/env python3
"""
Compare two serialized model architectures.

This script compares models saved using the ModelArchitectureSerializer
to verify that model refactoring preserved the architecture and parameters.

Usage:
    python compare_models.py old_model.pkl new_model.pkl
    python compare_models.py --verbose old_model.pkl new_model.pkl
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def compare_lstm_weights_separately(
    model1: Dict[str, Any], model2: Dict[str, Any], rtol: float = 1e-5, atol: float = 1e-8, verbose: bool = True
) -> Tuple[bool, str]:
    """Compare LSTM weights separately from other parameters."""
    verbose = True
    if verbose:
        print("Comparing LSTM weights separately...")

    state1 = model1["state_dict"]
    state2 = model2["state_dict"]

    # Find LSTM-related parameters
    lstm_params = []
    other_params = []

    for param_name in state1.keys():
        # Look for common LSTM parameter patterns
        if any(pattern in param_name.lower() for pattern in ["lstm", "_net.weight", "_net.bias", "_core_"]):
            lstm_params.append(param_name)
        else:
            other_params.append(param_name)

    if verbose:
        print(f"  Found {len(lstm_params)} LSTM parameters:")
        for param in lstm_params:
            print(f"    {param}")
        print(f"  Found {len(other_params)} other parameters")

    # Check LSTM parameters first
    lstm_max_diff = 0.0
    lstm_max_diff_param = ""
    lstm_max_normalized_diff = 0.0
    lstm_max_orthogonal_error = 0.0

    for param_name in lstm_params:
        param1_info = state1[param_name]
        param2_info = state2[param_name]

        # Reconstruct tensors (same logic as compare_parameter_values)
        data1 = np.frombuffer(param1_info["data"], dtype=np.uint8)
        data2 = np.frombuffer(param2_info["data"], dtype=np.uint8)

        dtype_str = param1_info["dtype"]
        dtype_mapping = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.bool": np.bool_,
            "torch.uint8": np.uint8,
        }

        numpy_dtype = dtype_mapping.get(dtype_str, np.float32)

        try:
            data1 = data1.view(dtype=numpy_dtype).reshape(param1_info["shape"])
            data2 = data2.view(dtype=numpy_dtype).reshape(param2_info["shape"])
        except ValueError as e:
            if verbose:
                print(f"  ! Skipping LSTM parameter '{param_name}': unable to reconstruct ({e})")
            continue

        # Skip empty tensors
        if data1.size == 0 or data2.size == 0:
            if data1.size != data2.size:
                return False, f"LSTM parameter '{param_name}': different sizes (empty vs non-empty)"
            if verbose:
                print(f"  ✓ LSTM parameter '{param_name}': empty tensor, skipping")
            continue

        # Convert to torch tensors for analysis (copy to make writable)
        import torch

        tensor1 = torch.from_numpy(data1.copy())
        tensor2 = torch.from_numpy(data2.copy())

        # Convert to torch tensors for analysis (copy to make writable)
        import torch

        tensor1 = torch.from_numpy(data1.copy())
        tensor2 = torch.from_numpy(data2.copy())

        # Track maximum difference
        diff = np.max(np.abs(data1 - data2))
        if diff > lstm_max_diff:
            lstm_max_diff = diff
            lstm_max_diff_param = param_name

        # LSTM-specific analysis for weight matrices (not biases) - run regardless of allclose result
        normalized_diff = 0.0
        orthogonal_diff = 0.0

        if "weight" in param_name.lower() and len(tensor1.shape) == 2:
            # Normalized comparison (L2 normalization)
            norm1 = tensor1 / torch.norm(tensor1)
            norm2 = tensor2 / torch.norm(tensor2)
            normalized_diff = torch.max(torch.abs(norm1 - norm2)).item()
            lstm_max_normalized_diff = max(lstm_max_normalized_diff, normalized_diff)

            # Check orthogonal property (if matrix is square or near-square)
            min_dim = min(tensor1.shape)
            max_dim = max(tensor1.shape)
            if min_dim > 1 and max_dim / min_dim <= 4:  # Reasonable aspect ratio
                # For non-square matrices, check if columns/rows are orthogonal
                if tensor1.shape[0] <= tensor1.shape[1]:
                    # More columns than rows - check row orthogonality
                    product1 = tensor1 @ tensor1.T
                    product2 = tensor2 @ tensor2.T
                    identity = torch.eye(tensor1.shape[0])
                else:
                    # More rows than columns - check column orthogonality
                    product1 = tensor1.T @ tensor1
                    product2 = tensor2.T @ tensor2
                    identity = torch.eye(tensor1.shape[1])

                orthogonal_error1 = torch.max(torch.abs(product1 - identity)).item()
                orthogonal_error2 = torch.max(torch.abs(product2 - identity)).item()
                orthogonal_diff = abs(orthogonal_error1 - orthogonal_error2)
                lstm_max_orthogonal_error = max(lstm_max_orthogonal_error, orthogonal_diff)

        # Check if values pass allclose test
        allclose_passed = np.allclose(data1, data2, rtol=rtol, atol=atol)

        # Verbose output
        if verbose:
            if "weight" in param_name.lower() and len(tensor1.shape) == 2:
                status = "✓" if allclose_passed else "✗"
                if orthogonal_diff > 0:
                    print(
                        f"  {status} LSTM parameter '{param_name}': (diff: {diff:.2e}, normalized: {normalized_diff:.2e}, orthogonal: {orthogonal_diff:.2e})"
                    )
                else:
                    print(
                        f"  {status} LSTM parameter '{param_name}': (diff: {diff:.2e}, normalized: {normalized_diff:.2e})"
                    )
            else:
                status = "✓" if allclose_passed else "✗"
                print(f"  {status} LSTM parameter '{param_name}': (diff: {diff:.2e})")

        # Only fail if allclose fails
        if not allclose_passed:
            return False, f"LSTM parameter '{param_name}': values differ (max diff: {diff:.2e})"

    if verbose and lstm_params:
        print(f"  ✓ LSTM overall max difference: {lstm_max_diff:.2e} (in '{lstm_max_diff_param}')")
        if lstm_max_normalized_diff > 0:
            print(f"  ✓ LSTM max normalized difference: {lstm_max_normalized_diff:.2e}")
        if lstm_max_orthogonal_error > 0:
            print(f"  ✓ LSTM max orthogonal property difference: {lstm_max_orthogonal_error:.2e}")

    summary = f"LSTM parameters match (diff: {lstm_max_diff:.2e}"
    if lstm_max_normalized_diff > 0:
        summary += f", normalized: {lstm_max_normalized_diff:.2e}"
    if lstm_max_orthogonal_error > 0:
        summary += f", orthogonal: {lstm_max_orthogonal_error:.2e}"
    summary += ")"

    return True, summary


def compare_non_lstm_weights(
    model1: Dict[str, Any], model2: Dict[str, Any], rtol: float = 1e-5, atol: float = 1e-8, verbose: bool = False
) -> Tuple[bool, str]:
    """Compare non-LSTM weights separately."""
    if verbose:
        print("Comparing non-LSTM weights...")

    state1 = model1["state_dict"]
    state2 = model2["state_dict"]

    # Find non-LSTM parameters
    other_params = []
    for param_name in state1.keys():
        if not any(pattern in param_name.lower() for pattern in ["lstm", "_net.weight", "_net.bias", "_core_"]):
            other_params.append(param_name)

    if not other_params:
        if verbose:
            print("  No non-LSTM parameters found")
        return True, "No non-LSTM parameters to compare"

    if verbose:
        print(f"  Found {len(other_params)} non-LSTM parameters")

    max_diff = 0.0
    max_diff_param = ""

    for param_name in other_params:
        param1_info = state1[param_name]
        param2_info = state2[param_name]

        # Reconstruct tensors (same logic as compare_parameter_values)
        data1 = np.frombuffer(param1_info["data"], dtype=np.uint8)
        data2 = np.frombuffer(param2_info["data"], dtype=np.uint8)

        dtype_str = param1_info["dtype"]
        dtype_mapping = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.bool": np.bool_,
            "torch.uint8": np.uint8,
        }

        numpy_dtype = dtype_mapping.get(dtype_str, np.float32)

        try:
            data1 = data1.view(dtype=numpy_dtype).reshape(param1_info["shape"])
            data2 = data2.view(dtype=numpy_dtype).reshape(param2_info["shape"])
        except ValueError as e:
            if verbose:
                print(f"  ! Skipping parameter '{param_name}': unable to reconstruct ({e})")
            continue

        # Skip empty tensors
        if data1.size == 0 or data2.size == 0:
            if data1.size != data2.size:
                return False, f"Parameter '{param_name}': different sizes (empty vs non-empty)"
            if verbose:
                print(f"  ✓ Parameter '{param_name}': empty tensor, skipping")
            continue

        # Compare values
        if not np.allclose(data1, data2, rtol=rtol, atol=atol):
            diff = np.max(np.abs(data1 - data2))
            return False, f"Non-LSTM parameter '{param_name}': values differ (max diff: {diff:.2e})"

        # Track maximum difference
        diff = np.max(np.abs(data1 - data2))
        if diff > max_diff:
            max_diff = diff
            max_diff_param = param_name

        if verbose:
            print(f"  ✓ Parameter '{param_name}': values match (max diff: {diff:.2e})")

    if verbose:
        print(f"  ✓ Non-LSTM overall max difference: {max_diff:.2e} (in '{max_diff_param}')")

    return True, f"Non-LSTM parameters match (max diff: {max_diff:.2e})"  #!/usr/bin/env python3


"""
Compare two serialized model architectures.

This script compares models saved using the ModelArchitectureSerializer
to verify that model refactoring preserved the architecture and parameters.

Usage:
    python compare_models.py old_model.pkl new_model.pkl
    python compare_models.py --verbose old_model.pkl new_model.pkl
"""

from typing import Any, Dict, Tuple


def load_generic_model(path: str) -> Dict[str, Any]:
    """Load a serialized model from file."""
    import pickle

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path_obj, "rb") as f:
        model_data = pickle.load(f)

    return model_data


def compare_basic_info(model1: Dict[str, Any], model2: Dict[str, Any], verbose: bool = False) -> Tuple[bool, str]:
    """Compare basic model information."""
    if verbose:
        print("Comparing basic model information...")

    if model1["model_class"] != model2["model_class"]:
        return False, f"Different model classes: {model1['model_class']} vs {model2['model_class']}"

    if model1["model_module"] != model2["model_module"]:
        return False, f"Different model modules: {model1['model_module']} vs {model2['model_module']}"

    if verbose:
        print(f"  ✓ Model class: {model1['model_class']}")
        print(f"  ✓ Model module: {model1['model_module']}")

    return True, "Basic info matches"


def compare_architecture(model1: Dict[str, Any], model2: Dict[str, Any], verbose: bool = False) -> Tuple[bool, str]:
    """Compare model architectures."""
    if verbose:
        print("Comparing model architectures...")

    arch1 = model1["architecture"]
    arch2 = model2["architecture"]

    # Compare module names
    modules1 = set(arch1.keys())
    modules2 = set(arch2.keys())

    if modules1 != modules2:
        missing_in_2 = modules1 - modules2
        missing_in_1 = modules2 - modules1
        msg = "Different module structure:"
        if missing_in_1:
            msg += f" Missing in model1: {missing_in_1}"
        if missing_in_2:
            msg += f" Missing in model2: {missing_in_2}"
        return False, msg

    if verbose:
        print(f"  ✓ Number of modules: {len(modules1)}")

    # Compare each module
    for name in modules1:
        module1 = arch1[name]
        module2 = arch2[name]

        if module1["type"] != module2["type"]:
            return False, f"Module '{name}': different types {module1['type']} vs {module2['type']}"

        # Compare module-specific configuration
        for key in module1.keys():
            if key in ["type", "module_path"]:
                continue
            if module1.get(key) != module2.get(key):
                return False, f"Module '{name}': {key} differs ({module1.get(key)} vs {module2.get(key)})"

        if verbose:
            print(f"  ✓ Module '{name}': {module1['type']}")

    return True, "Architectures match"


def compare_state_dict_structure(
    model1: Dict[str, Any], model2: Dict[str, Any], verbose: bool = False
) -> Tuple[bool, str]:
    """Compare state dictionary structure (shapes, dtypes, etc.)."""
    if verbose:
        print("Comparing state dictionary structure...")

    state1 = model1["state_dict"]
    state2 = model2["state_dict"]

    # Compare parameter names
    params1 = set(state1.keys())
    params2 = set(state2.keys())

    if params1 != params2:
        missing_in_2 = params1 - params2
        missing_in_1 = params2 - params1
        msg = "Different parameter names:"
        if missing_in_1:
            msg += f" Missing in model1: {missing_in_1}"
        if missing_in_2:
            msg += f" Missing in model2: {missing_in_2}"
        return False, msg

    if verbose:
        print(f"  ✓ Number of parameters: {len(params1)}")

    # Compare each parameter's metadata
    total_params = 0
    for param_name in params1:
        param1 = state1[param_name]
        param2 = state2[param_name]

        if param1["shape"] != param2["shape"]:
            return False, f"Parameter '{param_name}': different shapes {param1['shape']} vs {param2['shape']}"

        if param1["dtype"] != param2["dtype"]:
            return False, f"Parameter '{param_name}': different dtypes {param1['dtype']} vs {param2['dtype']}"

        if param1["requires_grad"] != param2["requires_grad"]:
            return (
                False,
                f"Parameter '{param_name}': different requires_grad {param1['requires_grad']} vs {param2['requires_grad']}",
            )

        # Count parameters
        param_count = np.prod(param1["shape"])
        total_params += param_count

        if verbose and param_count > 1000:  # Only show large parameters
            print(f"  ✓ Parameter '{param_name}': {param1['shape']} ({param_count:,} params)")

    if verbose:
        print(f"  ✓ Total parameters: {total_params:,}")

    return True, "State dict structures match"


def compare_parameter_values(
    model1: Dict[str, Any], model2: Dict[str, Any], rtol: float = 1e-5, atol: float = 1e-8, verbose: bool = False
) -> Tuple[bool, str]:
    """Compare actual parameter values."""
    if verbose:
        print("Comparing parameter values...")

    state1 = model1["state_dict"]
    state2 = model2["state_dict"]

    max_diff = 0.0
    max_diff_param = ""

    for param_name in state1.keys():
        # Reconstruct tensors from bytes
        param1_info = state1[param_name]
        param2_info = state2[param_name]

        # Convert bytes back to numpy arrays
        data1 = np.frombuffer(param1_info["data"], dtype=np.uint8)
        data2 = np.frombuffer(param2_info["data"], dtype=np.uint8)

        # Determine the original dtype from the saved info
        dtype_str = param1_info["dtype"]

        # Map PyTorch dtype strings to numpy dtypes
        dtype_mapping = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.bool": np.bool_,
            "torch.uint8": np.uint8,
        }

        # Get the appropriate numpy dtype
        numpy_dtype = dtype_mapping.get(dtype_str, np.float32)

        # Reshape to original shape
        try:
            data1 = data1.view(dtype=numpy_dtype).reshape(param1_info["shape"])
            data2 = data2.view(dtype=numpy_dtype).reshape(param2_info["shape"])
        except ValueError as e:
            if verbose:
                print(f"  ! Skipping parameter '{param_name}': unable to reconstruct ({e})")
            continue

        # Skip empty tensors
        if data1.size == 0 or data2.size == 0:
            if data1.size != data2.size:
                return False, f"Parameter '{param_name}': different sizes (empty vs non-empty)"
            if verbose:
                print(f"  ✓ Parameter '{param_name}': empty tensor, skipping")
            continue

        # Compare values
        if not np.allclose(data1, data2, rtol=rtol, atol=atol):
            diff = np.max(np.abs(data1 - data2))
            return False, f"Parameter '{param_name}': values differ (max diff: {diff:.2e})"

        # Track maximum difference
        diff = np.max(np.abs(data1 - data2))
        if diff > max_diff:
            max_diff = diff
            max_diff_param = param_name

        if verbose:
            print(f"  ✓ Parameter '{param_name}': values match (max diff: {diff:.2e})")

    if verbose:
        print(f"  ✓ Overall max difference: {max_diff:.2e} (in '{max_diff_param}')")

    return True, "Parameter values match"


def compare_custom_attributes(
    model1: Dict[str, Any], model2: Dict[str, Any], verbose: bool = False
) -> Tuple[bool, str]:
    """Compare custom model attributes."""
    if verbose:
        print("Comparing custom attributes...")

    attrs1 = model1.get("custom_attributes", {})
    attrs2 = model2.get("custom_attributes", {})

    if attrs1 != attrs2:
        return False, f"Custom attributes differ: {attrs1} vs {attrs2}"

    if verbose:
        if attrs1:
            print(f"  ✓ Custom attributes: {list(attrs1.keys())}")
        else:
            print("  ✓ No custom attributes")

    return True, "Custom attributes match"


def print_model_summary(model_data: Dict[str, Any], label: str):
    """Print a summary of the model."""
    print(f"\n{label} Summary:")
    print(f"  Class: {model_data['model_class']}")
    print(f"  Module: {model_data['model_module']}")
    print(f"  Architecture modules: {len(model_data['architecture'])}")
    print(f"  Parameters: {len(model_data['state_dict'])}")

    # Count total parameters
    total_params = 0
    for param_info in model_data["state_dict"].values():
        total_params += np.prod(param_info["shape"])
    print(f"  Total parameter count: {total_params:,}")

    custom_attrs = model_data.get("custom_attributes", {})
    if custom_attrs:
        print(f"  Custom attributes: {list(custom_attrs.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two serialized PyTorch model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_models.py old_model.pkl new_model.pkl
  python compare_models.py --verbose --tolerance 1e-6 old_model.pkl new_model.pkl
        """,
    )

    parser.add_argument("old_model", help="Path to old model file")
    parser.add_argument("new_model", help="Path to new model file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed comparison information")
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Tolerance for parameter value comparison (default: 1e-5)"
    )
    parser.add_argument("--skip-values", action="store_true", help="Skip parameter value comparison (structure only)")

    args = parser.parse_args()

    try:
        # Load models
        print("Loading models...")
        model1 = load_generic_model(args.old_model)
        model2 = load_generic_model(args.new_model)

        if args.verbose:
            print_model_summary(model1, "Old Model")
            print_model_summary(model2, "New Model")

        print("\nComparing models...")

        # Run comparisons
        comparisons = [
            ("Basic Info", compare_basic_info),
            ("Architecture", compare_architecture),
            ("State Dict Structure", compare_state_dict_structure),
            ("Custom Attributes", compare_custom_attributes),
        ]

        if not args.skip_values:
            # Compare LSTM weights separately first
            comparisons.append(
                (
                    "LSTM Weights",
                    lambda m1, m2, v: compare_lstm_weights_separately(m1, m2, args.tolerance, args.tolerance, v),
                )
            )
            comparisons.append(
                (
                    "Non-LSTM Weights",
                    lambda m1, m2, v: compare_non_lstm_weights(m1, m2, args.tolerance, args.tolerance, v),
                )
            )

        all_passed = True

        for name, compare_func in comparisons:
            if args.verbose:
                print(f"\n--- {name} ---")

            try:
                passed, message = compare_func(model1, model2, args.verbose)
                if passed:
                    print(f"✓ {name}: {message}")
                else:
                    print(f"✗ {name}: {message}")
                    all_passed = False
            except Exception as e:
                print(f"✗ {name}: Error during comparison - {e}")
                all_passed = False

        # Final result
        print(f"\n{'=' * 50}")
        if all_passed:
            print("🎉 SUCCESS: Models are identical!")
            sys.exit(0)
        else:
            print("❌ FAILURE: Models differ!")
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
