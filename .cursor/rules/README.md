# Cursor Rules (September 2025 - Simplified)

Streamlined Cursor IDE configuration with just 3 essential rule files for the Metta AI project.

## Simple Structure

### Just 3 Files
- **001_core.mdc** (Always) - Project context, commands, AI behavior (31 lines)
- **101_python.mdc** (Auto: `*.py`) - Python/ML standards (26 lines)  
- **201_dev.mdc** (Auto: tests/frontend) - Testing, frontend, planning (28 lines)

Total: **85 lines** (down from 153+ lines across 8 files)

## What Each File Covers

### 001_core.mdc (Always Applied)
- Metta AI project overview
- Essential commands (`uv run`, `metta test`, recipe system)
- AI behavioral guidelines
- Critical Python formatting requirement

### 101_python.mdc (Python Files)
- Import organization and type annotations
- Class design patterns (private/public members)
- ML/RL specific patterns (PolicyStore, MettaAgent, policy URIs)
- PyTorch device management

### 201_dev.mdc (Development Files)
- Testing commands and integration patterns
- Frontend development (TypeScript/React)
- Task planning with ExitPlanMode
- File reference format

## Benefits of Simplified Structure

- **Faster Loading**: Fewer files to process
- **Less Maintenance**: Single source for related concepts
- **Better Overview**: Easier to see all rules at a glance
- **Reduced Redundancy**: No duplicate information across files
- **Token Efficient**: All essential info in minimal space

## Usage Patterns

- **Core context** always available for all interactions
- **Python rules** activate when editing `.py` files
- **Dev tools** activate for tests, frontend, or planning tasks

This simplified structure maintains all essential guidance while being much more manageable and efficient.