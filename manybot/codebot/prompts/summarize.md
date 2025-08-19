# Code Snippet Selection Task

## Objective
Select and organize the most important code snippets from the provided files to create a structured summary optimized for AI consumption.

## Task Requirements

### 1. Component Identification
For each significant code component, extract:
- **Key function/class signatures** with essential parameters
- **Critical type definitions** and interfaces
- **Important constants** and configuration values
- **Entry point functions** and main execution paths

### 2. Dependency Mapping
Identify and include:
- **Import statements** that reveal external dependencies
- **Internal module relationships** through import patterns
- **Interface boundaries** between major components

### 3. Pattern Recognition
Extract code snippets that demonstrate:
- **Architectural patterns** (factories, decorators, inheritance hierarchies)
- **Error handling approaches** (try/catch blocks, error types)
- **Configuration patterns** (settings, environment variables)
- **Testing patterns** (fixture setup, mock usage)

## Selection Criteria

### Include These Code Snippets:
- Function signatures with docstrings
- Class definitions with key methods
- Type annotations and data models
- Configuration dictionaries/objects
- Main execution blocks (`if __name__ == "__main__"`)
- API endpoint definitions
- Database schema definitions
- Error handling patterns

### Exclude These:
- Implementation details within function bodies
- Verbose comments and documentation
- Test data and fixtures (unless they reveal usage patterns)
- Generated code or boilerplate

## Output Structure

Organize selected snippets by:

1. **Entry Points**: Main functions, CLI commands, API endpoints
2. **Core Types**: Key classes, data models, interfaces
3. **External Dependencies**: Import statements grouped by purpose
4. **Configuration**: Settings, constants, environment handling
5. **Patterns**: Recurring code structures and architectural decisions

## Token Efficiency

- Stay under {{MAX_TOKENS}} tokens total
- Prioritize **code snippets over prose**
- Use **minimal connecting text** between snippets
- **Truncate large functions** to signature + key logic only
- **Group similar patterns** to avoid repetition

Focus on providing the essential code structure that enables understanding and generation tasks, not explanatory text.
