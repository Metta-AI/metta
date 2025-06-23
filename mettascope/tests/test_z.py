# Reads a compressed JSON file and prints the contents.
import json
import zlib

# Read the compressed JSON file

file_name = "replay.json.z"

with open(file_name, "rb") as file:
    compressed_data = file.read()

# Decompress the data
decompressed_data = zlib.decompress(compressed_data)

# Parse the JSON data
json_data = json.loads(decompressed_data)

# Print the JSON data
print(json_data)
