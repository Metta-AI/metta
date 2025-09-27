"""Compatibility entrypoint for utilities maintained in ``softmax.lib``."""

from softmax.lib.utils import (  # re-export canonical helpers
    ParsedURI,
    calculate_batch_sizes,
    calculate_prioritized_sampling_params,
    exists,
    http_url,
    is_public_uri,
    local_copy,
    read,
    write_data,
    write_file,
)

__all__ = [
    "ParsedURI",
    "calculate_batch_sizes",
    "calculate_prioritized_sampling_params",
    "exists",
    "http_url",
    "is_public_uri",
    "local_copy",
    "read",
    "write_data",
    "write_file",
]
