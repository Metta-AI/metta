# MkDocs Documentation System

This document explains how MkDocs addresses the feedback about manual documentation maintenance and provides automatic generation of navigation and indices.

## Benefits Over Manual Approach

### 1. **No Manual Index Maintenance**
- Replaces `docs/index.md` with automatic search functionality
- Full-text search across all documentation
- No need to manually update when files are added/removed

### 2. **Automatic Table of Contents**
- Every page gets an automatic TOC generated from headers
- No need to maintain TOCs in markdown files
- Configurable depth and styling

### 3. **Dynamic Navigation**
- Navigation structure updates automatically
- Can auto-discover documentation files
- Single source of truth in `mkdocs.yml`

## Quick Start

```bash
# Install dependencies (already in pyproject.toml)
uv sync

# Serve documentation locally with live reload
uv run mkdocs serve

# Build static documentation site
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

## Key Features

### Search Instead of Index
- Built-in search replaces manual alphabetical index
- Searches content, not just titles
- Instant results with highlighting

### Auto-Generated Elements
1. **Navigation**: From directory structure
2. **Table of Contents**: From markdown headers
3. **API Docs**: From Python docstrings (with mkdocstrings)
4. **Cross-references**: Smart linking between pages

### Integration with Existing Docs
- Works with all existing markdown files
- No need to rewrite documentation
- Preserves current file structure

## Configuration

The `mkdocs.yml` file controls:
- Site metadata
- Theme and appearance
- Plugin configuration
- Navigation structure (with auto-discovery options)

## Migration from Manual System

1. **Remove Manual TOCs**: MkDocs generates these automatically
2. **Delete `docs/index.md`**: Replaced by search functionality
3. **Keep Everything Else**: All other markdown files work as-is

## Advanced Features

### Auto-Discovery Script
The `scripts/gen_docs_nav.py` script can:
- Find all README files automatically
- Build navigation from directory structure
- Generate indices dynamically

### GitHub Pages Integration
```yaml
# In .github/workflows/docs.yml
- name: Deploy docs
  run: |
    uv run mkdocs gh-deploy --force
```

This approach eliminates the need for manual documentation maintenance while providing better features than a static index file.
