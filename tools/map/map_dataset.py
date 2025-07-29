from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.map.load_maps import load_maps


class MapDataset(Dataset):
    """PyTorch Dataset for loading maps."""

    def __init__(self, file_path: str):
        self.maps_data: List[Dict[str, Any]] = load_maps(file_path)

    def __len__(self) -> int:
        return len(self.maps_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        map_item = self.maps_data[idx]
        map_lines = map_item["map"]

        # Remove border
        content_lines = [line[1:-1] for line in map_lines[1:-1]]

        height = len(content_lines)
        width = len(content_lines[0]) if height > 0 else 0

        # Convert to a numerical format (1 for wall, 0 for empty)
        # Shape: [height, width]
        grid = np.zeros((height, width), dtype=np.float32)
        for r, row_str in enumerate(content_lines):
            for c, char in enumerate(row_str):
                if char == "#":
                    grid[r, c] = 1.0

        # Add feature dimension for PyTorch (C, H, W)
        # Shape: [1, height, width]
        grid = np.expand_dims(grid, axis=0)

        return torch.from_numpy(grid)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create a PyTorch DataLoader for maps.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file containing map data.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the DataLoader.")
    args = parser.parse_args()

    dataset = MapDataset(args.file_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get a single batch
    try:
        first_batch = next(iter(dataloader))
        print("Successfully loaded a batch of maps.")
        print(f"Shape of the batch (B, C, H, W): {first_batch.shape}")

    except StopIteration:
        print("The dataset is empty or the file path is incorrect.")


if __name__ == "__main__":
    main()
