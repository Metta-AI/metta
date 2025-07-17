"""Generate documentation navigation automatically for MkDocs.

This script is used with the mkdocs-gen-files plugin to automatically
discover and add documentation files to the navigation, eliminating
the need for manual index maintenance.
"""

from pathlib import Path

try:
    import mkdocs_gen_files
except ImportError:
    # This module is only available when run by MkDocs
    print("Note: This script should be run by MkDocs, not directly")
    exit(0)

# This script would be used with mkdocs-gen-files plugin to automatically
# discover and add documentation files to the navigation

nav = mkdocs_gen_files.Nav()
root = Path(".")

# Auto-discover all README files in component directories
component_dirs = ["agent", "common", "mettagrid", "mettascope", "observatory", "app_backend"]
for component in component_dirs:
    readme_path = Path(component) / "README.md"
    if readme_path.exists():
        # Add to navigation automatically
        nav[component.title()] = str(readme_path)

# Auto-discover all markdown files in docs directory
for path in sorted(Path("docs").rglob("*.md")):
    # Skip the old manual index file
    if path.name == "index.md":
        continue

    # Build navigation hierarchy from path
    parts = path.relative_to("docs").parts
    nav[parts] = str(path)

# Generate a summary page with all discovered documentation
with mkdocs_gen_files.open("reference/auto_index.md", "w") as f:
    f.write("# Auto-Generated Documentation Index\n\n")
    f.write("This index is automatically generated from all documentation files.\n\n")

    # Write navigation as a structured list
    for item in nav.build_literate_nav():
        f.write(item)

# This replaces the need for manually maintaining docs/index.md!
