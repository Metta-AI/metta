"""
Checkpoint compression utilities for MettaAgent.

This module provides functions to compress and decompress checkpoints,
reducing storage requirements especially for save_for_training() checkpoints.
"""

import gzip
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import lz4.frame
import torch
import zstandard as zstd

logger = logging.getLogger(__name__)


class CheckpointCompressor:
    """Handles compression and decompression of checkpoint files."""

    COMPRESSION_METHODS = {
        "none": {"extension": "", "level": 0},
        "gzip": {"extension": ".gz", "level": 6},
        "lz4": {"extension": ".lz4", "level": 0},  # LZ4 doesn't use levels same way
        "zstd": {"extension": ".zst", "level": 3},
    }

    def __init__(self, method: str = "zstd", level: Optional[int] = None):
        """
        Initialize compressor with specified method.

        Args:
            method: Compression method ("none", "gzip", "lz4", "zstd")
            level: Compression level (method-specific, None for default)
        """
        if method not in self.COMPRESSION_METHODS:
            raise ValueError(f"Unknown compression method: {method}")

        self.method = method
        self.config = self.COMPRESSION_METHODS[method].copy()

        if level is not None:
            self.config["level"] = level

    def compress_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Compress a checkpoint file.

        Args:
            input_path: Path to uncompressed checkpoint
            output_path: Output path (if None, adds compression extension)

        Returns:
            Path to compressed file
        """
        if self.method == "none":
            if output_path and output_path != input_path:
                shutil.copy2(input_path, output_path)
                return output_path
            return input_path

        if output_path is None:
            output_path = input_path + self.config["extension"]

        logger.info(f"Compressing {input_path} with {self.method}")
        start_size = os.path.getsize(input_path)

        if self.method == "gzip":
            with open(input_path, "rb") as f_in:
                with gzip.open(output_path, "wb", compresslevel=self.config["level"]) as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif self.method == "lz4":
            with open(input_path, "rb") as f_in:
                with lz4.frame.open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif self.method == "zstd":
            cctx = zstd.ZstdCompressor(level=self.config["level"])
            with open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    cctx.copy_stream(f_in, f_out)

        end_size = os.path.getsize(output_path)
        ratio = (1 - end_size / start_size) * 100
        logger.info(f"Compressed {start_size} -> {end_size} bytes ({ratio:.1f}% reduction)")

        return output_path

    def decompress_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Decompress a checkpoint file.

        Args:
            input_path: Path to compressed checkpoint
            output_path: Output path (if None, removes compression extension)

        Returns:
            Path to decompressed file
        """
        # Detect compression method from extension
        method = self._detect_compression(input_path)

        if method == "none":
            if output_path and output_path != input_path:
                shutil.copy2(input_path, output_path)
                return output_path
            return input_path

        if output_path is None:
            # Remove compression extension
            for ext in [".gz", ".lz4", ".zst"]:
                if input_path.endswith(ext):
                    output_path = input_path[: -len(ext)]
                    break
            else:
                output_path = input_path + ".decompressed"

        logger.info(f"Decompressing {input_path} with {method}")

        if method == "gzip":
            with gzip.open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif method == "lz4":
            with lz4.frame.open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif method == "zstd":
            dctx = zstd.ZstdDecompressor()
            with open(input_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    dctx.copy_stream(f_in, f_out)

        return output_path

    def _detect_compression(self, path: str) -> str:
        """Detect compression method from file extension or content."""
        if path.endswith(".gz"):
            return "gzip"
        elif path.endswith(".lz4"):
            return "lz4"
        elif path.endswith(".zst"):
            return "zstd"

        # Try to detect from file header
        with open(path, "rb") as f:
            header = f.read(4)

        if header[:2] == b"\x1f\x8b":  # gzip magic number
            return "gzip"
        elif header == b"\x04\x22\x4d\x18":  # LZ4 magic number
            return "lz4"
        elif header[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic number
            return "zstd"

        return "none"

    def save_compressed(self, data: Any, path: str) -> str:
        """
        Save data with compression in one step.

        Args:
            data: Data to save (typically a dict)
            path: Output path (compression extension added if needed)

        Returns:
            Path to saved file
        """
        if not path.endswith(self.config["extension"]) and self.method != "none":
            path = path + self.config["extension"]

        if self.method == "none":
            torch.save(data, path)
        else:
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                torch.save(data, tmp.name)
                tmp_path = tmp.name

            try:
                # Compress the temporary file
                self.compress_file(tmp_path, path)
            finally:
                os.unlink(tmp_path)

        return path

    def load_compressed(self, path: str, map_location="cpu", weights_only=False) -> Any:
        """
        Load data with automatic decompression.

        Args:
            path: Path to potentially compressed file
            map_location: PyTorch map_location parameter
            weights_only: PyTorch weights_only parameter

        Returns:
            Loaded data
        """
        method = self._detect_compression(path)

        if method == "none":
            return torch.load(path, map_location=map_location, weights_only=weights_only)

        # Decompress to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp_path = tmp.name

        try:
            self.decompress_file(path, tmp_path)
            return torch.load(tmp_path, map_location=map_location, weights_only=weights_only)
        finally:
            os.unlink(tmp_path)


def compress_checkpoint_directory(
    directory: str, method: str = "zstd", level: Optional[int] = None, remove_originals: bool = False
) -> Dict[str, str]:
    """
    Compress all checkpoints in a directory.

    Args:
        directory: Directory containing .pt files
        method: Compression method
        level: Compression level
        remove_originals: Whether to delete original files

    Returns:
        Dict mapping original paths to compressed paths
    """
    compressor = CheckpointCompressor(method, level)
    results = {}

    for pt_file in Path(directory).glob("*.pt"):
        if any(str(pt_file).endswith(ext) for ext in [".gz", ".lz4", ".zst"]):
            continue  # Skip already compressed files

        compressed_path = compressor.compress_file(str(pt_file))
        results[str(pt_file)] = compressed_path

        if remove_originals and compressed_path != str(pt_file):
            os.unlink(pt_file)

    return results


def benchmark_compression_methods(checkpoint_path: str) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different compression methods on a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint to test

    Returns:
        Dict with compression stats for each method
    """
    import time

    results = {}
    original_size = os.path.getsize(checkpoint_path)

    for method in ["gzip", "lz4", "zstd"]:
        for level in [1, 3, 6, 9]:
            if method == "lz4" and level > 1:
                continue  # LZ4 doesn't have traditional levels

            key = f"{method}_L{level}"
            compressor = CheckpointCompressor(method, level)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "compressed")

                # Time compression
                start = time.time()
                compressed_path = compressor.compress_file(checkpoint_path, output_path)
                compress_time = time.time() - start

                compressed_size = os.path.getsize(compressed_path)

                # Time decompression
                start = time.time()
                compressor.decompress_file(compressed_path, os.path.join(tmpdir, "decompressed"))
                decompress_time = time.time() - start

            results[key] = {
                "compress_time": compress_time,
                "decompress_time": decompress_time,
                "compressed_size": compressed_size,
                "original_size": original_size,
                "ratio": compressed_size / original_size,
                "reduction": (1 - compressed_size / original_size) * 100,
            }

            logger.info(
                f"{key}: {results[key]['reduction']:.1f}% reduction, "
                f"compress: {compress_time:.2f}s, decompress: {decompress_time:.2f}s"
            )

    return results


# Convenience functions for common use cases
def save_compressed(data: Any, path: str, method: str = "zstd") -> str:
    """Save checkpoint with compression."""
    compressor = CheckpointCompressor(method)
    return compressor.save_compressed(data, path)


def load_compressed(path: str, map_location="cpu", weights_only=False) -> Any:
    """Load checkpoint with automatic decompression."""
    compressor = CheckpointCompressor()
    return compressor.load_compressed(path, map_location, weights_only)
