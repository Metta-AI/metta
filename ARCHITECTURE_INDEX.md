# Metta AI Architecture Documentation Index

Welcome to the Metta AI architecture documentation. This index will help you navigate the comprehensive architectural maps created for this codebase.

## Documents Available

### 1. ARCHITECTURE_QUICK_REFERENCE.md (324 lines, 8.5KB)
**Best for**: Quick lookups, fast understanding, command reference

Use this when you need:
- Quick overview of the system
- Command examples and cheatsheet
- Directory structure reference
- Policy architecture comparison
- Troubleshooting table
- Key files quick lookup

**Start here if**: You have 10-15 minutes and want a high-level overview

**Navigation**: Flat document, use Ctrl+F to find topics

---

### 2. ARCHITECTURE.md (1,177 lines, 37KB)
**Best for**: Deep understanding, implementation details, comprehensive reference

This document has 7 major sections plus extras:

#### Section 1: Overview
- High-level architecture overview
- Core system components
- Data flow from recipes through execution

**Use when**: Getting oriented or explaining architecture to others

#### Section 2: Core Components & Their Responsibilities (10 subsections)
1. **Main Entry Point** (tools/run.py)
   - User-facing CLI interface
   - Recipe discovery and tool loading
   - File: `/Users/mp/Metta-ai/metta/tools/run.py`

2. **Tool Classes** (TrainTool, EvaluateTool, etc.)
   - Configuration classes for operations
   - Files: `/Users/mp/Metta-ai/metta/metta/tools/`

3. **Trainer** (Core Training Loop)
   - Main training orchestrator
   - Pluggable components architecture
   - File: `/Users/mp/Metta-ai/metta/metta/rl/trainer.py`

4. **CheckpointManager** (Policy Persistence)
   - Policy loading/saving with URI addressing
   - File: `/Users/mp/Metta-ai/metta/metta/rl/checkpoint_manager.py`

5. **Policies** (Neural Networks)
   - Policy interface and implementations
   - Files: `/Users/mp/Metta-ai/metta/agent/src/metta/agent/`

6. **Training Environment** (Experience Collection)
   - Vectorized environment wrapper
   - File: `/Users/mp/Metta-ai/metta/metta/rl/training/training_environment.py`

7. **Simulation** (Evaluation Infrastructure)
   - Policy evaluation runner
   - File: `/Users/mp/Metta-ai/metta/metta/sim/simulation.py`

8. **Loss Functions** (Training Objectives)
   - PPO, GRPO, Contrastive, etc.
   - Directory: `/Users/mp/Metta-ai/metta/metta/rl/loss/`

9. **MettaGrid** (Simulation Core)
   - C++ environment with Python bindings
   - Directory: `/Users/mp/Metta-ai/metta/packages/mettagrid/`

10. **Statistics & Analysis** (Metrics)
    - DuckDB-based storage and analysis
    - Directory: `/Users/mp/Metta-ai/metta/metta/sim/stats/`

**Use when**: Understanding how specific components work

#### Section 3: Recipe System
- How recipes bundle configurations
- Prod vs Experiment recipes
- Validation suites (CI and stable)
- Recipe discovery mechanism
- Code examples

**Use when**: Creating or understanding recipes

#### Section 4: Data Flow
- Training pipeline (7 steps)
- Evaluation pipeline (5 steps)
- Component communication patterns
- Policy URIs and ComponentContext

**Use when**: Tracing how data moves through the system

#### Section 5: Supporting Infrastructure
- Visualization tools (MettaScope, Observatory, GridWorks)
- Configuration system (OmegaConf)
- Analysis tools (DuckDB, notebooks)
- Distributed training
- Monitoring and profiling

**Use when**: Working with infrastructure components

#### Section 6: Key Entry Points
- User-facing CLI (tools/run.py)
- Programmatic recipe functions
- Job configuration
- Direct training (advanced)

**Use when**: Learning how to invoke the system

#### Section 7: Development Workflows
- Training models (with examples)
- Evaluating policies
- Interactive testing
- Creating new recipes
- CI/validation testing
- Code quality

**Use when**: Following step-by-step instructions

#### Additional Sections
- **Architecture Decisions & Trade-offs**: Design rationale
- **File Organization Summary**: Directory structure with paths
- **Key Patterns & Best Practices**: Code examples
- **Troubleshooting Guide**: Common issues and solutions
- **Resources**: Links to key documentation

---

## Quick Navigation Guide

### I want to...

**Understand the big picture**
1. Read Overview section in ARCHITECTURE.md
2. Look at data flow diagrams in Section 4
3. Review key entry points in Section 6

**Train a model**
1. Check "Training a Model" in Development Workflows
2. Reference command examples in ARCHITECTURE_QUICK_REFERENCE.md
3. Follow Training Pipeline in Data Flow section

**Evaluate a policy**
1. Check "Evaluating a Policy" in Development Workflows
2. Review Evaluation Pipeline in Data Flow section
3. Understand Simulation component in Core Components

**Understand the recipe system**
1. Read entire Recipe System section (Section 3)
2. See code examples with full recipe structure
3. Understand validation suites (CI vs Stable)

**Create a new recipe**
1. "Creating a New Recipe" in Development Workflows
2. Full Recipe System section for context
3. Example from `recipes/prod/arena_basic_easy_shaped.py`

**Add a new policy architecture**
1. "Key Patterns & Best Practices" section
2. Review Policies section in Core Components
3. Look at existing implementations in `agent/src/metta/agent/policies/`

**Add a new loss function**
1. "Extension Points" in ARCHITECTURE_QUICK_REFERENCE.md
2. Loss Functions section in Core Components
3. Look at `metta/rl/loss/*.py` for examples

**Debug an issue**
1. Check Troubleshooting Guide sections
2. Search ARCHITECTURE.md with Ctrl+F
3. Follow file paths to implementation

**Understand distributed training**
1. Find "Distributed Training" in Supporting Infrastructure
2. Review DistributedHelper implementation
3. Check Training component interactions

**Analyze training results**
1. Find "Analysis Infrastructure" in Supporting Infrastructure
2. Review Statistics & Analysis component
3. Check development workflow examples

---

## Document Features

### Both Documents Include:
- Clear section hierarchies
- Absolute file paths for easy navigation
- Code examples and patterns
- Command examples
- Architecture diagrams (text-based)

### ARCHITECTURE.md Unique Features:
- Comprehensive coverage of every system
- Detailed component responsibilities
- Complete data flow documentation
- Architecture decision rationale
- Best practices and patterns

### ARCHITECTURE_QUICK_REFERENCE.md Unique Features:
- Quick lookup tables
- Command cheatsheet
- At-a-glance summaries
- Minimal reading for fast answers

---

## How to Use These Documents

### For New Team Members (First Day)
1. Start with ARCHITECTURE_QUICK_REFERENCE.md (read entirely, ~20 min)
2. Run the examples to get familiar with commands
3. Jump to ARCHITECTURE.md sections as needed for deeper understanding
4. Follow file paths to actual code

### For Implementation Work
1. Find relevant section in ARCHITECTURE.md
2. Review code files listed in that section
3. Check "Extension Points" for customization
4. Reference "Key Patterns" for implementation examples

### For Code Review
1. Use data flow sections to understand impact
2. Reference architecture decisions for context
3. Use component sections to verify design compliance
4. Check patterns for consistency

### For Troubleshooting
1. Check Troubleshooting Guide sections
2. Search for keywords in both documents
3. Follow file paths to relevant code
4. Review similar working examples

---

## File Locations

All documentation is in the Metta AI repository root:

```
/Users/mp/Metta-ai/metta/
├── ARCHITECTURE_INDEX.md          # This file
├── ARCHITECTURE.md                # Comprehensive reference (1,177 lines)
├── ARCHITECTURE_QUICK_REFERENCE.md # Quick lookup (324 lines)
├── CLAUDE.md                      # Development guidelines
├── README.md                      # Project overview
└── [rest of repository...]
```

---

## Search Tips

### In ARCHITECTURE.md
Use Ctrl+F (Cmd+F on Mac) to search for:
- Component names: "Trainer", "CheckpointManager", "Simulation"
- Concepts: "pipeline", "data flow", "recipe"
- File paths: Search for filenames to find locations
- Section titles: Jump directly to topics

### In ARCHITECTURE_QUICK_REFERENCE.md
Use for:
- Quick command examples
- File path quick reference
- Configuration options
- Troubleshooting steps

---

## Cross-References

### From ARCHITECTURE.md to Code
Every major component section lists exact file paths:
- File paths are absolute
- Can be opened directly in editor
- Includes all related files in each component section

### From ARCHITECTURE_QUICK_REFERENCE.md to Detailed Docs
Quick reference sections link to ARCHITECTURE.md sections:
- "See Data Flow section in ARCHITECTURE.md"
- "See Core Components section for details"
- Easy navigation between documents

---

## Keeping Documentation Updated

These documents were created by thorough exploration of the codebase on 2025-11-14.

To update after codebase changes:
1. Follow the same exploration pattern
2. Update relevant sections
3. Add new components if added to codebase
4. Update file paths if structure changes
5. Add new recipes or tools to examples

---

## Questions or Improvements?

If you find:
- Errors or inaccuracies - update the relevant section
- Missing information - add new sections following the pattern
- Better examples - replace with improved versions
- Clearer explanations - refactor for clarity

Both documents are designed to be living documentation.

---

## Related Resources

- **README.md**: Project overview and quick start guide
- **CLAUDE.md**: Development guidelines and best practices
- **roadmap.md**: Research directions and future work
- **Individual component READMEs**: 
  - `agent/README.md` - Policy architectures
  - `packages/mettagrid/README.md` - Environment details
  - `common/src/metta/common/tool/README.md` - Tool runner details
  - `observatory/README.md` - Dashboard documentation
  - `gridworks/README.md` - Web interface

---

Start reading: [ARCHITECTURE_QUICK_REFERENCE.md](ARCHITECTURE_QUICK_REFERENCE.md) or [ARCHITECTURE.md](ARCHITECTURE.md)
