import json
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_json_file>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, "r") as f:
    layers = json.load(f)

# Sum all sizes
total_size = sum(layer["size"] for layer in layers)
print("Total size:", total_size)
