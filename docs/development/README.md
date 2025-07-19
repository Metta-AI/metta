# Development Guidelines

This directory contains guidelines and standards for developers and AI tools working on the Metta AI codebase.

## 📚 Available Guidelines

### [AI Agent Guidelines](./agents.md)
Guidelines for Codex agents and other AI assistants working in this repository:
- Commit message standards
- Quality check requirements
- Type annotation rules
- Pull request documentation

### [Claude Development Guide](./claude.md)
Comprehensive guidance for Claude AI when working with the codebase:
- Project overview and architecture
- Development environment setup
- Code style guidelines
- PR creation standards
- Configuration system overview
- Testing philosophy

## 🔧 Key Standards

### Code Quality
- Run `ruff format` and `ruff check` before committing
- Add type annotations to all function parameters
- Follow selective return type annotation guidelines
- Use modern Python typing syntax (PEP 585)

### Documentation
- Keep commit messages short and in present tense
- Include test output in PR descriptions
- Document complex logic with clear comments
- Update relevant documentation when making changes

### Testing
- Run tests with `uv run pytest` or `pytest`
- Ensure tests are independent and idempotent
- Cover edge cases and boundary conditions
- Mirror project structure in test organization

## 🚀 Getting Started

1. Read the appropriate guide for your development context
2. Set up your development environment with `./install.sh`
3. Configure your editor to follow project standards
4. Run quality checks before submitting changes

## 📝 Contributing

When updating these guidelines:
- Ensure consistency across documents
- Update examples to reflect current best practices
- Consider impact on existing workflows
- Test any new procedures before documenting them
