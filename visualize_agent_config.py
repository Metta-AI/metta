#!/usr/bin/env python
import argparse
import os
from textwrap import dedent

import graphviz
import yaml

# Define standard component names for environment interaction
ENV_NODE_NAME = "Environment (mettagrid)"
OBS_INPUT_COMPONENT = "_obs_"
ACTION_TYPE_COMPONENT = "_action_type_"
ACTION_PARAM_COMPONENT = "_action_param_"

# Map target class names to descriptions
COMPONENT_DESCRIPTIONS = {
    "ObsShaper": "Reshapes/Selects raw env observation data.",
    "ObservationNormalizer": "Normalizes observation values (e.g., using running mean/std).",
    "Conv2d": "Extracts spatial features via 2D convolution (filters).",
    "Flatten": "Reshapes multi-dimensional feature maps into a 1D vector.",
    "Linear": "Applies affine transformation (fully connected layer).",
    "LSTM": "Processes sequence, updates hidden state (temporal memory).",
    "ActionType": "Outputs logits for discrete action categories.",
    "ActionParam": "Outputs parameters for the selected action type.",
    # Add more descriptions as needed for other component types
}


def format_value(value):
    """Helper to format values, especially dictionaries like nn_params."""
    if isinstance(value, dict):
        # Format dictionary nicely over multiple lines
        items = [f"  {k}: {v}" for k, v in value.items()]
        return "\n".join(["{"] + items + ["}"])
    return str(value)


def visualize_config(yaml_path, output_filename=None, view=False):
    """
    Parses a MettaAgent YAML config and generates a Graphviz visualization.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        output_filename (str, optional): Name for the output file (without extension).
                                         If None, defaults to {config_name}_agent_config_graph.
        view (bool): If True, attempts to open the generated graph automatically.
    """
    # Determine default output filename if not provided
    if output_filename is None:
        base_name = os.path.basename(yaml_path)
        config_name, _ = os.path.splitext(base_name)
        output_filename = f"{config_name}_agent_config_graph"

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {yaml_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")
        return

    if "components" not in config or not isinstance(config["components"], dict):
        print(f"Error: 'components' key not found or is not a dictionary in {yaml_path}")
        return

    dot = graphviz.Digraph(comment="MettaAgent Architecture with Environment", format="png")
    dot.attr(rankdir="TB")  # Top to bottom layout

    components = config["components"]

    # Add Environment Node
    dot.node(ENV_NODE_NAME, shape="ellipse", style="filled", color="lightblue")

    # Add agent component nodes with details
    for name, params in components.items():
        if not isinstance(params, dict):
            print(f"Warning: Skipping component '{name}' as its value is not a dictionary.")
            continue

        label_lines = [f"<{name}>", "--------"]
        target_class_full = params.get("_target_", "N/A")
        target_class_short = target_class_full.split(".")[-1]  # Get short class name
        label_lines.append(f"_target_: {target_class_short}")

        # Add description based on target class
        description = COMPONENT_DESCRIPTIONS.get(target_class_short, "(Unknown function)")
        label_lines.append(f"Desc: {description}")

        # Add other relevant parameters to the label
        for key, value in params.items():
            if key not in ["_target_", "input_source"]:
                # Use format_value to handle nested dicts like nn_params
                label_lines.append(f"{key}: {format_value(value)}")

        label = "\n".join(label_lines)
        dot.node(name, label=label, shape="box")

    # Add internal edges based on input_source
    for name, params in components.items():
        if not isinstance(params, dict):
            continue  # Already warned above
        input_source = params.get("input_source")
        if input_source:
            # Define edge label based on source/target (best effort)
            edge_label = f"Data from {input_source}"  # Default label
            source_target = components.get(input_source, {}).get("_target_", "").split(".")[-1]
            target_target = params.get("_target_", "").split(".")[-1]

            if source_target == "ObservationNormalizer" and target_target == "Conv2d":
                edge_label = "Normalized Obs"
            elif source_target == "Conv2d" and target_target == "Conv2d":
                edge_label = "Feature Maps"
            elif source_target == "Conv2d" and target_target == "Flatten":
                edge_label = "Feature Maps"
            elif source_target == "Flatten" and target_target == "Linear":
                edge_label = "Flattened Features"
            elif source_target == "Linear" and target_target == "Linear":
                edge_label = "Latent Vector"
            elif source_target == "Linear" and target_target == "LSTM":
                edge_label = "Encoded Obs Input"
            elif source_target == "LSTM":
                edge_label = "LSTM Hidden State"
            elif source_target == "Linear" and target_target == "ActionType":
                edge_label = "Action Pathway Features"
            elif source_target == "Linear" and target_target == "ActionParam":
                edge_label = "Action Pathway Features"
            elif (
                source_target == "Linear" and target_target == "Linear" and name == "_value_"
            ):  # Specific case for value output
                edge_label = "Value Pathway Features"
            elif source_target == "ObsShaper" and target_target == "ObservationNormalizer":
                edge_label = "Shaped Obs"
            # Add more specific rules if needed

            # Handle potential list of sources if ever needed, currently assumes single string
            if isinstance(input_source, str):
                if input_source in components:
                    dot.edge(input_source, name, label=edge_label)
                else:
                    print(f"Warning: Input source '{input_source}' for component '{name}' not found in components.")
            elif isinstance(input_source, list):
                for source in input_source:
                    if source in components:
                        dot.edge(source, name, label=edge_label)  # Apply same label for multi-input for simplicity
                    else:
                        print(f"Warning: Input source '{source}' for component '{name}' not found in components.")
            else:
                print(f"Warning: Invalid input_source format for component '{name}': {input_source}")

    # Add edges for Environment Interaction
    # Observation Input Edge
    if OBS_INPUT_COMPONENT in components:
        dot.edge(ENV_NODE_NAME, OBS_INPUT_COMPONENT, label="Raw Observations")
    else:
        print(f"Warning: Standard observation input component '{OBS_INPUT_COMPONENT}' not found.")

    # Action Output Edges
    action_outputs_found = False
    if ACTION_TYPE_COMPONENT in components:
        dot.edge(ACTION_TYPE_COMPONENT, ENV_NODE_NAME, label="Action Type Selection")
        action_outputs_found = True
    if ACTION_PARAM_COMPONENT in components:
        dot.edge(ACTION_PARAM_COMPONENT, ENV_NODE_NAME, label="Action Parameters")
        action_outputs_found = True

    if not action_outputs_found:
        print(
            f"Warning: Standard action output components ('{ACTION_TYPE_COMPONENT}', '{ACTION_PARAM_COMPONENT}') not found."
        )

    # Render the graph
    try:
        output_path = os.path.abspath(output_filename)
        dot.render(output_path, view=view, cleanup=True)
        print(f"Graph saved to {output_path}.png")
    except graphviz.backend.execute.ExecutableNotFound:
        print(
            dedent(f"""
            Error: Graphviz executable not found. Please install Graphviz.
            Instructions: https://graphviz.org/download/
            Alternatively, if using pip: pip install graphviz
            If using conda: conda install python-graphviz
            Graph source saved to {output_path}
        """)
        )
    except Exception as e:
        print(f"Error rendering graph: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MettaAgent configuration from a YAML file.")
    parser.add_argument("config_file", type=str, help="Path to the agent YAML configuration file.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename (without extension). Default: {config_name}_agent_config_graph",
    )
    parser.add_argument("--view", action="store_true", help="Attempt to open the generated graph.")

    args = parser.parse_args()

    visualize_config(args.config_file, args.output, args.view)
