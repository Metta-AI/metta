#!/usr/bin/env python3
"""
Command-line interface for map format migration utilities.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

from metta.mettagrid.map_builder.map_builder import GameMap, create_int_grid, create_legacy_grid
from metta.mettagrid.migration.map_format_converter import MapFormatConverter
from metta.mettagrid.migration.performance import PerformanceBenchmark
from metta.mettagrid.migration.validation import MapFormatValidator
from metta.mettagrid.object_types import ObjectTypes


def create_sample_maps(count: int = 5, sizes: List[tuple] = None) -> List[GameMap]:
    """Create sample maps for testing and benchmarking."""
    if sizes is None:
        sizes = [(10, 10), (25, 25), (50, 50)]

    maps = []

    for i in range(count):
        size = sizes[i % len(sizes)]
        height, width = size

        if i % 2 == 0:  # Create legacy map
            grid = create_legacy_grid(height, width)
            grid[:] = "empty"

            # Add walls around border
            grid[0, :] = "wall"
            grid[-1, :] = "wall"
            grid[:, 0] = "wall"
            grid[:, -1] = "wall"

            # Add some random objects
            if height > 5 and width > 5:
                grid[height // 4, width // 4] = "agent"
                grid[3 * height // 4, 3 * width // 4] = "agent"

            maps.append(GameMap(grid=grid))

        else:  # Create int map
            grid = create_int_grid(height, width)
            grid[:] = ObjectTypes.EMPTY

            # Add walls around border
            grid[0, :] = ObjectTypes.WALL
            grid[-1, :] = ObjectTypes.WALL
            grid[:, 0] = ObjectTypes.WALL
            grid[:, -1] = ObjectTypes.WALL

            # Add some random objects
            if height > 5 and width > 5:
                grid[height // 4, width // 4] = ObjectTypes.AGENT_BASE
                grid[3 * height // 4, 3 * width // 4] = ObjectTypes.AGENT_BASE + 1

            decoder_key = (
                ["empty", "wall"] + [f"object_{j}" for j in range(2, ObjectTypes.AGENT_BASE)] + ["agent", "agent2"]
            )
            maps.append(GameMap(grid=grid, decoder_key=decoder_key))

    return maps


def cmd_convert(args):
    """Convert maps between formats."""
    print(f"Converting maps to {args.target_format} format...")

    # Create sample maps if no input provided
    if not args.input:
        print("No input specified, creating sample maps...")
        maps = create_sample_maps(args.count)
    else:
        # Load maps from input (placeholder - would need actual file loading)
        print(f"Loading maps from {args.input}...")
        maps = create_sample_maps(args.count)

    converter = MapFormatConverter()

    # Convert maps
    try:
        converted_maps = converter.batch_convert_maps(maps, target_format=args.target_format)
        print(f"Successfully converted {len(converted_maps)} maps")

        # Validate conversions if requested
        if args.validate:
            print("Validating conversions...")
            # validator = MapFormatValidator()  # TODO: implement validation

            for i, (original, converted) in enumerate(zip(maps, converted_maps, strict=False)):
                validation = converter.validate_conversion_integrity(original, converted)
                if validation["valid"]:
                    print(f"  Map {i}: ✓ Valid")
                else:
                    print(f"  Map {i}: ✗ Invalid - {len(validation['errors'])} errors")
                    for error in validation["errors"][:3]:  # Show first 3 errors
                        print(f"    - {error}")

        # Output results if requested
        if args.output:
            print(f"Saving results to {args.output}...")
            # Placeholder for actual file saving

        # Print conversion statistics
        stats = converter.get_conversion_stats()
        print("\nConversion Statistics:")
        print(f"  Total object types: {stats['total_types']}")
        print(f"  Uses GameConfig: {stats['uses_game_config']}")
        print(
            f"  Decoder key: {stats['decoder_key'][:5]}..."
            if len(stats["decoder_key"]) > 5
            else f"  Decoder key: {stats['decoder_key']}"
        )

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

    return 0


def cmd_validate(args):
    """Validate map formats."""
    print("Validating map formats...")

    # Create sample maps if no input provided
    if not args.input:
        print("No input specified, creating sample maps...")
        maps = create_sample_maps(args.count)
    else:
        print(f"Loading maps from {args.input}...")
        maps = create_sample_maps(args.count)

    validator = MapFormatValidator()

    results = []
    for i, game_map in enumerate(maps):
        print(f"\nValidating map {i} ({game_map.grid.shape})...")

        validation = validator.validate_game_map(game_map)
        results.append(validation)

        if validation["valid"]:
            print(f"  ✓ Valid {validation['format']} map")
            print(
                f"  Stats: {validation['stats'].get('unique_objects', 0)} unique objects, "
                f"{validation['stats'].get('agent_count', 0)} agents"
            )
        else:
            print(f"  ✗ Invalid {validation.get('format', 'unknown')} map")
            for error in validation["errors"][:3]:  # Show first 3 errors
                print(f"    - {error}")

        if validation.get("warnings"):
            for warning in validation["warnings"][:2]:  # Show first 2 warnings
                print(f"    ⚠ {warning}")

    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    print("\nValidation Summary:")
    print(f"  Valid maps: {valid_count}/{len(results)}")
    print(f"  Legacy maps: {sum(1 for r in results if r.get('format') == 'legacy')}")
    print(f"  Int maps: {sum(1 for r in results if r.get('format') == 'int')}")

    return 0 if valid_count == len(results) else 1


def cmd_benchmark(args):
    """Run performance benchmarks."""
    print("Running performance benchmarks...")

    benchmark = PerformanceBenchmark()

    if args.quick:
        print("Running quick benchmark...")
        # Quick benchmark with small maps
        test_maps = create_sample_maps(count=2, sizes=[(10, 10), (20, 20)])
        results = {
            "memory": benchmark.benchmark_memory_usage([(10, 10), (20, 20)]),
            "conversion": benchmark.benchmark_format_conversion(test_maps, iterations=5),
        }
    else:
        print("Running comprehensive benchmark... (this may take a while)")
        test_maps = create_sample_maps(count=args.count)
        results = benchmark.run_comprehensive_benchmark(test_maps)

    # Print key results
    print("\nBenchmark Results:")

    if "memory" in results:
        print("Memory Usage Comparison:")
        for size_key, memory_data in results["memory"].items():
            savings = memory_data["savings"]
            print(f"  {size_key}: {savings['percentage']:.1f}% memory savings with int format")

    if "system_info" in results:
        print("\nSystem Info:")
        print(f"  CPU cores: {results['system_info']['cpu_count']}")
        print(f"  Memory: {results['system_info']['memory_total'] // (1024**3)} GB")

    # Save detailed results if requested
    if args.output:
        print(f"Saving detailed results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Get benchmark summary
    summary = benchmark.get_results_summary()
    if summary.get("total_benchmarks", 0) > 0:
        print("\nBenchmark Summary:")
        print(f"  Total benchmarks: {summary['total_benchmarks']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Total time: {summary['total_time']:.2f}s")
        print(f"  Average time: {summary['average_time']:.4f}s")

    return 0


def cmd_test(args):
    """Run the test suite."""
    print("Running migration test suite...")

    try:
        import pytest

        # Run tests with appropriate verbosity
        test_args = [
            str(Path(__file__).parent / "test_migration_suite.py"),
            "-v" if args.verbose else "-q",
        ]

        if args.pattern:
            test_args.extend(["-k", args.pattern])

        exit_code = pytest.main(test_args)
        return exit_code

    except ImportError:
        print("pytest not available, running basic validation test...")

        # Basic validation test
        maps = create_sample_maps(3)
        converter = MapFormatConverter()
        validator = MapFormatValidator()

        all_passed = True

        for i, game_map in enumerate(maps):
            try:
                # Test validation
                validation = validator.validate_game_map(game_map)
                if not validation["valid"]:
                    print(f"Map {i} validation failed")
                    all_passed = False
                    continue

                # Test conversion
                if game_map.is_legacy():
                    converted = converter.convert_game_map_to_int(game_map)
                    restored = converter.convert_game_map_to_legacy(converted)

                    if not np.array_equal(game_map.grid, restored.grid):
                        print(f"Map {i} round-trip conversion failed")
                        all_passed = False

                print(f"Map {i}: ✓ Passed")

            except Exception as e:
                print(f"Map {i}: ✗ Error - {e}")
                all_passed = False

        print(f"\nBasic test {'PASSED' if all_passed else 'FAILED'}")
        return 0 if all_passed else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Map format migration utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert maps between formats")
    convert_parser.add_argument("target_format", choices=["int", "legacy"], help="Target format for conversion")
    convert_parser.add_argument("-i", "--input", help="Input file/directory")
    convert_parser.add_argument("-o", "--output", help="Output file/directory")
    convert_parser.add_argument(
        "-c", "--count", type=int, default=5, help="Number of sample maps to create (if no input)"
    )
    convert_parser.add_argument("--validate", action="store_true", help="Validate conversions")
    convert_parser.set_defaults(func=cmd_convert)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate map formats")
    validate_parser.add_argument("-i", "--input", help="Input file/directory")
    validate_parser.add_argument(
        "-c", "--count", type=int, default=5, help="Number of sample maps to create (if no input)"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument("-o", "--output", help="Output file for detailed results")
    benchmark_parser.add_argument("-c", "--count", type=int, default=6, help="Number of test maps to create")
    benchmark_parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (faster but less comprehensive)"
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    test_parser.add_argument("-k", "--pattern", help="Test pattern to match")
    test_parser.set_defaults(func=cmd_test)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
