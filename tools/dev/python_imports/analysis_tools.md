# Python Import Refactoring Tools

Analysis tools for identifying and fixing Python import issues.

## Quick Start

### Detect Circular Dependencies

```bash
python tools/dev/python_imports/detect_cycles.py --path . --output cycles_report.json
```

Finds:

- Circular import dependencies (real runtime cycles)
- TYPE_CHECKING block usage
- Local imports (style violations)

### Analyze Architecture

```bash
python tools/dev/python_imports/analyze_architecture.py --path . --output architecture_report.json
```

Recommends:

- Types to extract to `types.py` files
- `__init__.py` files to simplify
- Lazy-loading patterns to refactor

## Documentation

- **SPECIFICATION.md** - Formal rules for import patterns
