"""
Token profiling functionality with integrated flame graph generation and proper common prefix handling
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken

from .file import get_context


@dataclass
class TokenNode:
    """Node for tracking token counts in a hierarchy."""

    label: str
    parent: Optional["TokenNode"]
    total_tokens: int = 0

    @property
    def pct_of_parent(self) -> float:
        if not self.parent:
            return 100.0
        return (self.total_tokens / self.parent.total_tokens * 100) if self.parent.total_tokens else 0.0

    @property
    def pct_of_total(self) -> float:
        root = self
        while root.parent:
            root = root.parent
        return (self.total_tokens / root.total_tokens * 100) if root.total_tokens else 0.0


class TokenProfiler:
    """Profiles token distribution across files and directories."""

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        # Cache nodes by full path
        self.node_cache: Dict[str, TokenNode] = {}
        # Track file types
        self.type_counts: Dict[str, int] = {}
        self.total_tokens = 0
        # Store paths for common prefix calculation
        self.file_paths: List[str] = []
        # Track individual file tokens
        self.file_tokens: Dict[str, int] = {}

    def count_tokens(self, content: str) -> int:
        """Count tokens in content using configured encoding."""
        return len(self.encoding.encode(content))

    def get_or_create_node(self, path: Path) -> TokenNode:
        """Get existing node or create new one with proper hierarchy."""
        str_path = str(path)

        # Store the path for common prefix calculation
        self.file_paths.append(str_path)

        if str_path in self.node_cache:
            return self.node_cache[str_path]

        # Create parent first if needed
        parent = None
        if path.parent != path:
            parent = self.get_or_create_node(path.parent)

        node = TokenNode(path.name, parent)
        self.node_cache[str_path] = node
        return node

    def process_file(self, filepath: Path, content: str, token_count: Optional[int] = None) -> None:
        """Process single file and update relevant counts."""
        # Use provided token count or calculate it
        if token_count is None:
            token_count = self.count_tokens(content)

        # Track individual file tokens
        self.file_tokens[str(filepath)] = token_count

        # Update directory hierarchy
        node = self.get_or_create_node(filepath.parent)
        current = node
        while current:
            current.total_tokens += token_count
            current = current.parent

        # Also create a node for the file itself
        file_node = self.get_or_create_node(filepath)
        file_node.total_tokens = token_count

        # Update file type stats
        ext = filepath.suffix.lower()
        self.type_counts[ext] = self.type_counts.get(ext, 0) + token_count
        self.total_tokens += token_count

    def format_hierarchical_report(self, common_prefix: str, requested_paths: Optional[List[Path]] = None) -> str:
        """Generate hierarchical token distribution report.

        Args:
            common_prefix: Common path prefix to remove from display
            requested_paths: Original paths that were requested (to filter out from display)
        """
        lines = ["Token Distribution by Directory:"]
        lines.append("-" * 80)
        lines.append(f"{'Directory':<50} {'Tokens':>10} {'% Parent':>10} {'% Total':>10}")
        lines.append("-" * 80)

        # Find common prefix to exclude from display
        prefix_len = len(common_prefix) if common_prefix else 0

        # Sort by path depth then alphabetically
        paths = sorted(self.node_cache.keys(), key=lambda x: (len(Path(x).parts), x))

        # Convert requested paths to absolute for comparison
        abs_requested_paths = {str(p.resolve()) for p in (requested_paths or [])}

        # Find the project root to use as base for display
        for path in paths:
            node = self.node_cache[path]
            path_obj = Path(path)

            # Skip the path if it's the common prefix or shorter
            if path == common_prefix or len(path) < prefix_len:
                continue

            # Determine if this is a file or directory
            is_file = path in self.file_tokens

            # For directories, check if they should be displayed
            if not is_file:
                path_obj_abs = str(path_obj.resolve())

                # If only one path requested, hide the requested path itself
                if len(abs_requested_paths) == 1 and path_obj_abs in abs_requested_paths:
                    continue

                # Check if this directory should be shown
                should_show = False

                # For multiple requested paths, show the requested paths themselves
                if len(abs_requested_paths) > 1 and path_obj_abs in abs_requested_paths:
                    should_show = True
                else:
                    # Check if this is a child of any requested path
                    for req_path in abs_requested_paths:
                        try:
                            # Check if this directory is a child of a requested path
                            Path(path_obj_abs).relative_to(req_path)
                            # Make sure it's not the requested path itself
                            if path_obj_abs != req_path:
                                should_show = True
                                break
                        except ValueError:
                            # Not a child of this requested path
                            continue

                # Skip directories that shouldn't be shown
                if not should_show:
                    continue

            # Remove common prefix for display
            display_path = path[prefix_len:] if prefix_len > 0 else path

            # Calculate indent based on path depth
            # For files, we need to count the full path including the file itself
            # For directories, we don't include the directory name in the count
            parts = [p for p in display_path.split(os.sep) if p]
            if is_file:
                # Files should be indented one level more than their parent directory
                indent = "  " * len(parts)
            else:
                # Directories are indented based on their depth
                indent = "  " * max(0, len(parts) - 1)

            # Use the last part of the path as the name
            name = path_obj.name or display_path

            lines.append(
                f"{indent}{name:<50} {node.total_tokens:>10} {node.pct_of_parent:>9.1f}% {node.pct_of_total:>9.1f}%"
            )

        return "\n".join(lines)

    def format_type_report(self) -> str:
        """Generate file type distribution report."""
        lines = ["Token Distribution by File Type:"]
        lines.append("-" * 60)
        lines.append(f"{'Type':<20} {'Tokens':>10} {'% Total':>10}")
        lines.append("-" * 60)

        # Sort by token count descending
        sorted_types = sorted(self.type_counts.items(), key=lambda x: x[1], reverse=True)

        for ext, count in sorted_types:
            ext_name = ext if ext else "(no extension)"
            pct = count / self.total_tokens * 100 if self.total_tokens else 0
            lines.append(f"{ext_name:<20} {count:>10} {pct:>9.1f}%")

        return "\n".join(lines)


def extract_files_from_context(context: str) -> Dict[str, str]:
    """Extract file paths and contents from context string."""
    files = {}

    # Parse XML format
    pattern = (
        r"<document .*?>\s*<source>(.*?)</source>.*?"
        r"(?:<document_content>(.*?)</document_content>|<instructions>(.*?)</instructions>).*?</document>"
    )
    matches = re.findall(pattern, context, re.DOTALL)
    for match in matches:
        path = match[0].strip()
        content = match[1] if match[1] else match[2]
        files[path] = content

    return files


def profile_code_context(
    paths: List[Union[str, Path]],
    extensions: Optional[Tuple[str, ...]] = None,
    include_git_diff: bool = False,
    readmes_only: bool = False,
    ignore_dirs: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, Dict]:
    """
    Profile token distribution for the given paths.

    Args:
        paths: List of paths to profile
        extensions: Optional file extensions to filter by
        include_git_diff: Whether to include git diff as a virtual file
        readmes_only: Whether to only include README.md files
        ignore_dirs: Optional tuple of directories to ignore

    Returns:
        Tuple of (formatted report, profile data)
    """
    # Get the context and token info first
    context, token_info = get_context(
        paths=paths,
        extensions=extensions,
        include_git_diff=include_git_diff,
        diff_base="origin/main",
        readmes_only=readmes_only,
        ignore_dirs=ignore_dirs,
    )

    # Use documents from token_info
    documents = token_info.get("documents", [])
    if not documents:
        message = "No files found to profile. Check your paths and extensions."
        return (message, {})

    # Build files dict from documents
    files = {doc.source: doc.content for doc in documents}

    # Profile the files using token counts from get_context
    profiler = TokenProfiler()
    file_paths = list(files.keys())

    # Get per-file token counts from token_info to avoid re-tokenization
    file_token_counts = token_info.get("file_token_counts", {})

    # Resolve requested paths
    from .file import resolve_codebase_path

    requested_paths = [resolve_codebase_path(p) for p in paths]

    # Process files using pre-calculated token counts
    for filepath, content in files.items():
        # Use pre-calculated token count if available
        token_count = file_token_counts.get(filepath)
        profiler.process_file(Path(filepath), content, token_count=token_count)

    # Find common path prefix for the title
    common_prefix = os.path.commonprefix(file_paths)
    base_dir = Path(common_prefix).name if common_prefix else "/"

    # Summary statistics
    total_tokens = profiler.total_tokens
    total_files = len(files)

    summary = [
        f"{base_dir} Token Profile Summary:",
        f"Total Files: {total_files}",
        f"Total Tokens: {total_tokens:,}",
        f"Avg Tokens per File: {total_tokens / total_files:.1f}",
        f"Common Path Prefix: {common_prefix}",
        "",
    ]

    # Resolve requested paths for filtering
    from .file import resolve_codebase_path

    requested_paths = [resolve_codebase_path(p) for p in paths]

    report = (
        "\n".join(summary)
        + "\n\n"
        + "\n\n".join(
            [profiler.format_hierarchical_report(common_prefix, requested_paths), profiler.format_type_report()]
        )
    )

    # Return both the formatted report and the profiler data
    profile_data = {
        "node_cache": profiler.node_cache,
        "type_counts": profiler.type_counts,
        "total_tokens": profiler.total_tokens,
        "total_files": total_files,
        "common_prefix": common_prefix,
        "file_paths": profiler.file_paths,
        "file_tokens": profiler.file_tokens,
        "context": context,  # Include the content we already retrieved
        "token_info": token_info,  # Include basic token info from get_context
    }
    return report, profile_data


def generate_flamegraph(data: Dict[str, Any], output_path: str, title: str = "Flame Graph", width: int = 1200) -> None:
    """
    Generate an interactive flame graph visualization from hierarchical data.

    Args:
        data: Dictionary with node_cache, total_tokens and other profile data
        output_path: Path to save the HTML file
        title: Title for the visualization
        width: Width of the flame graph in pixels
    """
    # Extract node cache and common prefix from profile data
    node_cache = data.get("node_cache", {})
    total_tokens = data.get("total_tokens", 0)
    common_prefix = data.get("common_prefix", "")

    logger = logging.getLogger("metta.setup.tools.code.token_profiler")
    logger.debug(f"Using common prefix: {common_prefix}")

    # Convert node cache to hierarchical dictionary with trimmed paths
    flame_dict = {}

    # Group nodes by parent path to build the hierarchy
    for path, node in node_cache.items():
        # Skip empty paths
        path_str = str(path)
        if not path_str:
            continue

        # Remove common prefix
        if common_prefix:
            if len(common_prefix) > len(path_str):
                continue
            elif path_str.startswith(common_prefix):
                path_str = path_str[len(common_prefix) :]
            else:
                logger.debug(f"Path {path_str} doesn't start with common prefix {common_prefix}")

        # Skip if empty after prefix removal
        if not path_str:
            continue

        # Split path
        path_parts = [part for part in path_str.split("/") if part]
        if not path_parts:
            continue

        current_level = flame_dict

        # Build the nested dictionary structure
        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1:
                # Leaf node - store the token count
                current_level[part] = node.total_tokens
            else:
                # Directory node - create/navigate to nested dict
                if part not in current_level:
                    current_level[part] = {}
                elif not isinstance(current_level[part], dict):
                    # If it's not a dict, make it one with a special _value key
                    value = current_level[part]
                    current_level[part] = {"_value": value}

                current_level = current_level[part]

    # Update title with token count and directory name
    base_dir = Path(common_prefix).name if common_prefix else "Project"
    full_title = f"{base_dir} Token Distribution - Total: {total_tokens:,} tokens"

    # Generate the HTML
    html = _generate_flamegraph_html(flame_dict, full_title, width)

    # Write to file
    with open(output_path, "w") as f:
        f.write(html)


def _generate_flamegraph_html(data_dict: Dict[str, Any], title: str = "Flame Graph", width: int = 1200) -> str:
    """
    Generate HTML with D3 flame graph from hierarchical dictionary.

    Args:
        data_dict: Hierarchical dictionary of token counts
        title: Title for the visualization
        width: Width of the flame graph in pixels

    Returns:
        HTML string for the flame graph visualization
    """

    # Convert nested dictionary to a d3 hierarchy
    def dict_to_hierarchy(d, name="root"):
        result = {"name": name, "children": []}

        for key, value in d.items():
            if isinstance(value, dict):
                if "_value" in value:
                    child = {"name": key, "value": value["_value"]}
                    result["children"].append(child)

                    # Add remaining items as children
                    sub_dict = {k: v for k, v in value.items() if k != "_value"}
                    if sub_dict:
                        child["children"] = dict_to_hierarchy(sub_dict, key)["children"]
                else:
                    child = dict_to_hierarchy(value, key)
                    result["children"].append(child)
            else:
                result["children"].append({"name": key, "value": value})

        return result

    # Convert to hierarchy
    hierarchy_data = dict_to_hierarchy(data_dict)

    # Generate HTML with embedded data
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
        }}
        #chart {{
            width: {width}px;
            margin-bottom: 20px;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        button {{
            padding: 8px 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 8px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .d3-flame-graph-tip {{
            line-height: 1.2;
            padding: 12px;
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            border-radius: 4px;
            pointer-events: none;
            z-index: 1000;
        }}
    </style>
    <!-- Using the exact CDN references that are known to work -->
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/spiermar/d3-flame-graph@2.0.3/dist/d3-flamegraph.css">
</head>
<body>
    <h1>{title}</h1>
    <div class="controls">
        <button id="resetZoom">Reset Zoom</button>
        <button id="toggleColors">Toggle Colors</button>
        <button id="toggleSort">Toggle Sort</button>
    </div>
    <div id="chart"></div>

    <!-- Using the exact CDN references that are known to work -->
    <script type="text/javascript" src="https://d3js.org/d3.v4.min.js"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3-tip/0.9.1/d3-tip.min.js"></script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/gh/spiermar/d3-flame-graph@2.0.3/dist/d3-flamegraph.min.js"></script>

    <script type="text/javascript">
        // Set up the flame graph
        var flamegraph = d3.flamegraph()
            .width({width})
            .cellHeight(20)
            .transitionDuration(750)
            .minFrameSize(5)
            .sort(true);

        // Create a custom color mapper for depth
        function depthColorMapper(d) {{
            // Color based on depth
            var colorsByDepth = [
                "#718dbf", // blue
                "#e49444", // orange
                "#d1615d", // red
                "#85b6b2", // teal
                "#6d8bba", // blue-purple
                "#8e91b9", // purple
                "#fe9ea8", // pink
                "#b3dcab", // light green
                "#c1c1c1"  // gray
            ];

            var depth = d.depth || 0;
            var colorIndex = depth % colorsByDepth.length;
            return colorsByDepth[colorIndex];
        }}

        // Apply the custom color mapper
        flamegraph.color(depthColorMapper);

        // Use a very simple tooltip function that just shows the name
        // This avoids issues with accessing values in different formats
        flamegraph.tooltip(function(d) {{
            return d.name;
        }});

        // Load the data and render
        var data = {json.dumps(hierarchy_data)};
        d3.select("#chart")
            .datum(data)
            .call(flamegraph);

        // Reset zoom button
        document.getElementById("resetZoom").addEventListener("click", function() {{
            flamegraph.resetZoom();
        }});

        // Toggle color schemes
        var currentColorScheme = "depth";
        document.getElementById("toggleColors").addEventListener("click", function() {{
            if (currentColorScheme === "depth") {{
                // Switch to a simpler color scheme
                flamegraph.color(function(d) {{
                    // Alternate colors based on depth
                    return d.depth % 2 ? "#a1afc3" : "#7894cb";
                }});
                currentColorScheme = "simple";
            }} else {{
                flamegraph.color(depthColorMapper);
                currentColorScheme = "depth";
            }}

            // Redraw with new colors
            d3.select("#chart")
                .datum(data)
                .call(flamegraph);
        }});

        // Toggle sorting
        var sorted = true;
        document.getElementById("toggleSort").addEventListener("click", function() {{
            sorted = !sorted;
            flamegraph.sort(sorted);

            // Redraw with new sort order
            d3.select("#chart")
                .datum(data)
                .call(flamegraph);
        }});
    </script>
</body>
</html>
    """

    return html
