#!/usr/bin/env python3
import argparse
import json
import random
import sys
import zlib

# Set up argument parser
parser = argparse.ArgumentParser(description="Decompress and preview JSON.z files")
parser.add_argument("filename", help="Input filename (*.json.z)")
parser.add_argument("-o", "--output", help="Output filename (defaults to input filename without .z)")
parser.add_argument("-m", "--modify", action="store_true", help="Modify action_names and add glyphs to type 0 objects")

args = parser.parse_args()

# Determine output filename
if args.output:
    output_file = args.output
else:
    # Remove .z extension if present
    if args.filename.endswith(".z"):
        output_file = args.filename[:-2]
    else:
        output_file = args.filename + ".json"

try:
    # Read the compressed JSON file
    with open(args.filename, "rb") as file:
        compressed_data = file.read()

    # Decompress the data
    decompressed_data = zlib.decompress(compressed_data)

    # Parse the JSON data
    json_data = json.loads(decompressed_data)

    # Modify if requested
    if args.modify:
        # Modify action_names
        if "action_names" in json_data:
            json_data["action_names"] = [
                "attack",
                "get_items",
                "move",
                "noop",
                "put_items",
                "rotate",
                "swap",
                "change_glyph",
            ]
            print("Modified action_names to include 'change_glyph'")
        else:
            print("Warning: 'action_names' key not found in JSON root")

        # Modify grid_objects
        if "grid_objects" in json_data:
            modified_count = 0
            for i, obj in enumerate(json_data["grid_objects"]):
                if obj.get("type") == 0:
                    arg = random.randint(0, 10)
                    obj["glyph"] = arg
                    obj["action"][0] = [0, [7, arg]]
                    obj["action_success"][0] = [0, True]
                    modified_count += 1

            print(
                f"Added 'glyph' to {modified_count} objects with type 0 (out of {len(json_data['grid_objects'])} total objects)"
            )
        else:
            print("Warning: 'grid_objects' key not found in JSON root")

    # Convert back to string
    json_string = json.dumps(json_data, separators=(",", ":"))

    # Print the first 1000 characters
    print("\nPreview (first 1000 characters):")
    print("-" * 40)
    print(json_string[:1000])
    print("-" * 40)
    print(f"Total length: {len(json_string)} characters")

    # Write to output file
    with open(output_file, "w") as out_file:
        out_file.write(json_string)

    print(f"\nDecompressed JSON written to: {output_file}")

except FileNotFoundError:
    print(f"Error: File '{args.filename}' not found")
    sys.exit(1)
except zlib.error as e:
    print(f"Error: Failed to decompress file - {e}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
