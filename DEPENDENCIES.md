# Dependency Management Guide

This project uses **Renovate** for automated dependency management with uv workspace support.

## Quick Reference

### Configuration
- **Config**: `.github/renovate.json5`
- **CI Validation**: `.github/workflows/dependency-validation.yml`
- **Schedule**: Weekend updates
- **Dashboard**: Check GitHub Issues for "Dependency Dashboard"

### Common Commands

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv add package_name@latest

# Check for conflicts
uv sync --frozen --check

# Validate workspace consistency
uv sync --all-packages
```

### Package Groups

Renovate groups related packages to reduce PR noise:

| Group | Packages |
|-------|----------|
| **pytest** | `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark` |
| **scientific** | `numpy`, `scipy`, `pandas`, `matplotlib`, `torch` |
| **rl** | `gymnasium`, `pettingzoo`, `shimmy`, `pufferlib` |
| **web** | `fastapi`, `uvicorn`, `starlette`, `pydantic` |
| **dev-tools** | `ruff`, `pyright`, `black`, `isort` |
| **cloud** | `boto3`, `botocore`, `google-*` |
| **jupyter** | `jupyter*`, `notebook`, `ipywidgets` |

## Troubleshooting

### Version Conflicts
1. Run `uv lock --upgrade` to resolve
2. Check `uv sync --frozen --check` for validation
3. Review conflicting packages in CI logs

### Workspace Issues
1. Ensure all packages use compatible versions
2. Run dependency validation CI locally
3. Check for duplicate dependency declarations

### Security Updates
- Renovate handles vulnerability alerts automatically
- Critical security updates override normal scheduling
- Review security advisories in PR descriptions

## Migration Notes

**⚠️ This project migrated from Dependabot to Renovate**
- Dependabot configuration removed entirely
- Renovate provides better uv workspace support
- More flexible grouping and auto-merge rules
- Enhanced CI validation pipeline

For detailed information, see the [Dependency Management section in CLAUDE.md](./CLAUDE.md#dependency-management).