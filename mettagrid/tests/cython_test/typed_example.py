"""Example of using array_processor with type annotations."""

from typing import List

import numpy as np
import numpy.typing as npt
from cython_processor import PyArrayProcessor


def process_batches(data_batches: List[npt.NDArray[np.uint8]], batch_size: int) -> List[npt.NDArray[np.uint8]]:
    """
    Process multiple batches of data.

    Args:
        data_batches: List of uint8 arrays to process
        batch_size: Size of each batch

    Returns:
        List of processed arrays
    """
    # Create processor with the right size
    processor = PyArrayProcessor(batch_size)

    # Process each batch
    results: List[npt.NDArray[np.uint8]] = []
    for batch in data_batches:
        # The IDE and type checker now know the return type
        processed: npt.NDArray[np.uint8] = processor.process_preallocated(batch)
        results.append(processed)

    return results


def main() -> None:
    # Create some test data
    batches = [np.array([1, 2, 3, 4, 5], dtype=np.uint8), np.array([10, 20, 30, 40, 50], dtype=np.uint8)]

    # Process batches
    results = process_batches(batches, 5)

    # Print results - IDE shows these are NDArray[np.uint8]
    for i, result in enumerate(results):
        print(f"Batch {i} result: {result}")


if __name__ == "__main__":
    main()
