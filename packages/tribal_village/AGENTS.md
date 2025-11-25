# Tribal Village Guidelines

See root `CLAUDE.md` and `STYLE_GUIDE.md` for general guidance.

## Build Commands

```bash
nimble install                    # Install Nim deps
nim r -d:release tribal_village.nim   # Run standalone
nimble buildLib                   # Build shared lib for Python
```

## Style

- Nim: 2-space indent, `lowerCamelCase` procs, `PascalCase` types
- Python: PEP 8 + type hints
