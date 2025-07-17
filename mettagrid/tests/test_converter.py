#!/usr/bin/env python3
"""Test that converters can show recipe inputs in observations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from metta.mettagrid.mettagrid_c import ConverterConfig


def test_converter_recipe_observations():
    """Test that ConverterConfig accepts show_recipe_inputs parameter."""

    # Create a converter with recipe display enabled
    converter_with_recipe = ConverterConfig(
        type_id=10,
        type_name="smelter_with_recipe",
        input_resources={0: 2, 1: 1},  # 2 ore, 1 fuel
        output_resources={2: 1},  # 1 metal
        max_output=5,
        conversion_ticks=10,
        cooldown=5,
        initial_resource_count=0,
        color=100,
        show_recipe_inputs=True,
    )

    # Create a converter without recipe display
    converter_no_recipe = ConverterConfig(
        type_id=11,
        type_name="smelter_no_recipe",
        input_resources={0: 2, 1: 1},  # 2 ore, 1 fuel
        output_resources={2: 1},  # 1 metal
        max_output=5,
        conversion_ticks=10,
        cooldown=5,
        initial_resource_count=0,
        color=101,
        show_recipe_inputs=False,
    )

    # Verify the property is set correctly
    assert converter_with_recipe.show_recipe_inputs, "show_recipe_inputs should be True"
    assert not converter_no_recipe.show_recipe_inputs, "show_recipe_inputs should be False"

    print("ConverterConfig properly supports show_recipe_inputs parameter")
    print(f"Converter 1 - show_recipe_inputs: {converter_with_recipe.show_recipe_inputs}")
    print(f"Converter 2 - show_recipe_inputs: {converter_no_recipe.show_recipe_inputs}")

    # Also test default value
    converter_default = ConverterConfig(
        type_id=12,
        type_name="smelter_default",
        input_resources={0: 1},
        output_resources={2: 1},
        max_output=5,
        conversion_ticks=10,
        cooldown=5,
    )

    assert not converter_default.show_recipe_inputs, "show_recipe_inputs should default to False"
    print(f"Converter default - show_recipe_inputs: {converter_default.show_recipe_inputs}")

    print("\nTest passed! Converters correctly support recipe input configuration.")


if __name__ == "__main__":
    test_converter_recipe_observations()
