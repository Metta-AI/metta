# Metta AI Documentation

Welcome to the Metta AI documentation hub. This directory contains comprehensive documentation for the project, organized into the following sections:

## 📖 Core Documentation

### [Quick Guide](./quick-guide.md)
A concise guide to get started with Metta AI in minutes. Perfect for newcomers who want to:
- Install and set up quickly
- Run their first training
- Visualize agents
- Understand key concepts

### [API Reference](./api.md)
Complete reference for the Metta API (`metta.api`), which provides a clean interface for using Metta's training components without Hydra configuration files. Includes examples for:
- Environment and agent creation
- Training loops and optimization
- Distributed training setup
- Checkpointing and recovery

### [Map Generation Guide](./mapgen.md)
Documentation for creating and working with procedurally generated maps:
- Creating maps in bulk with S3 storage
- Viewing and loading maps
- Using `LoadRandom` for dynamic map selection
- Integration with map_builder configs

## 🤖 Automation & Workflows

### [Workflows Documentation](./workflows/)
Detailed documentation for GitHub Actions workflows and automation:

- **[PR Summary System](./workflows/pr-summary.md)** - Automated weekly PR summaries using Gemini AI
- **[Claude Review System](./workflows/claude-review-system.md)** - AI-powered code review with targeted feedback
- **[Claude Assistant](./workflows/claude-assistant.md)** - Interactive bot for Q&A and automated PR creation

## 🛠️ Development Guidelines

### [Development Documentation](./development/)
Guidelines for developers and AI tools working on the codebase:

- **[AI Agent Guidelines](./development/agents.md)** - Guidelines for Codex agents working in the repository
- **[Claude Development Guide](./development/claude.md)** - Comprehensive guidance for Claude AI including project patterns and PR creation

## 📊 Metrics & Monitoring

### [WandB Metrics Guide](./wandb/metrics/)
Comprehensive documentation for all Weights & Biases metrics:
- Training metrics (losses, timing, memory usage)
- Environment metrics (agent stats, task completion, rewards)
- Evaluation metrics (navigation, object use, sequences)
- Performance profiling and monitoring

## 🔗 Quick Links

- **[Main README](../README.md)** - Project overview and quick start
- **[Research Roadmap](./roadmap.md)** - Detailed research directions
- **[Documentation Index](./index.md)** - Alphabetical index of all concepts
- **[Configuration Guide](../configs/)** - Configuration system reference

## 📝 Contributing to Documentation

When adding new documentation:

1. **Location**: Place topic-specific docs in this directory
2. **Format**: Use clear markdown with proper headers and code examples
3. **Linking**: Update this README and the index.md
4. **Standards**: Follow the patterns in existing documentation

### Documentation Structure

```
docs/
├── README.md           # This file
├── index.md           # Alphabetical index of concepts
├── roadmap.md         # Research roadmap
├── quick-guide.md     # Quick start guide
├── api.md             # API reference
├── mapgen.md          # Map generation guide
├── development/       # Development guidelines
│   ├── agents.md      # AI agent guidelines
│   └── claude.md      # Claude development guide
├── workflows/         # GitHub Actions documentation
│   ├── README.md      # Workflows overview
│   ├── pr-summary.md
│   ├── claude-review-system.md
│   └── claude-assistant.md
└── wandb/
    └── metrics/       # WandB metrics documentation
        ├── README.md
        └── [metric categories]/
```

## 🚀 Getting Started

New to Metta AI? Start here:

1. Read the [main README](../README.md) for project overview
2. Follow the [installation guide](../README.md#installation)
3. Check the [API documentation](./api.md) for programmatic usage
4. Explore [example configurations](../configs/)

For researchers:
- Review the [research roadmap](./roadmap.md)
- Explore [evaluation suites](../configs/eval/)
- Check [training configurations](../configs/trainer/)

For developers:
- Set up your [development environment](../README.md#development-setup)
- Review [testing guidelines](../tests/README.md)
- Check [workflow documentation](./workflows/) for CI/CD
