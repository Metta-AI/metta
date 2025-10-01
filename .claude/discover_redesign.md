# discover.py Redesign Proposal

## Current Issues

### Complexity (344 lines)
- **`get_available_tools()`** (40 lines): Complex introspection to find tools in modules using `dir()`, `inspect.isclass()`, `inspect.isfunction()`, signature parsing
- **`_function_returns_tool_type()`** (22 lines): Checks function return type annotations
- **`list_recipes_supporting_tool()`** (52 lines): Walks ALL recipe packages using `pkgutil.walk_packages()`
- **`generate_candidate_paths()`** (47 lines): Complex alias expansion with nested loops
- **`try_infer_tool_factory()`** (40 lines): Auto-factory with multiple fallback paths

### Unnecessary Metaprogramming
1. **Dynamic module walking** - `pkgutil.walk_packages()` to find all recipes
2. **Signature inspection** - Checking return type annotations to identify tools
3. **`dir()` reflection** - Iterating module attributes looking for tools
4. **Complex alias expansion** - Multi-level expansion logic

### Hidden Behavior
- Magic auto-discovery of tools in modules
- Implicit alias resolution in multiple places
- Recipe inference happens in multiple code paths

## Simplified Design (215 lines, -37% code)

### Core Principles
1. **Explicit > Implicit**: Registry knows all tools, no magic discovery
2. **Simple path resolution**: User input → candidate paths → try loading
3. **Clear responsibilities**: Each function does one thing

### Key Functions

```python
# Path resolution (simple string manipulation)
resolve_tool_path(user_input, second_token) → list[str]

# Tool loading (no introspection)
load_tool_maker(path) → Callable | None

# Recipe integration (cached)
get_recipe_configs(module) → (MettaGridConfig?, SimulationConfig[]?)
infer_tool_from_recipe(module_path, tool_name) → Callable | None

# Main entry
resolve_and_load_tool(user_input, second_token) → (path?, maker?)
```

### What's Gone
- ❌ `get_available_tools()` - No longer walk module attributes
- ❌ `_function_returns_tool_type()` - No signature inspection
- ❌ `list_recipes_supporting_tool()` - No package walking (can add back if needed)
- ❌ Complex alias expansion - Simplified to direct registry lookups
- ❌ Multiple helper functions - Consolidated logic

### What's Better
- ✅ 37% less code (344 → 215 lines)
- ✅ No `dir()`, `inspect.signature()`, or `pkgutil.walk_packages()`
- ✅ Clear flow: resolve → load → infer
- ✅ Single source of truth: ToolRegistry
- ✅ Same caching, better performance

## Migration Path

### Phase 1: Add new module
- Create `discover_simple.py` alongside existing
- Update `run_tool.py` to use new module
- Run all tests

### Phase 2: Remove old code
- Delete old `discover.py`
- Rename `discover_simple.py` → `discover.py`

### Phase 3: Feature parity (if needed)
- Re-add `--list` implementations support (simplified version)
- Can iterate without complex introspection

## Trade-offs

### Lost Features
- `list_recipes_supporting_tool()` - Can't show all recipes supporting a tool
  - **Impact**: `./tools/run.py train --help` won't list all train implementations
  - **Solution**: Can re-add with explicit recipe registry if needed

### Gained Benefits
- Much easier to understand and maintain
- No hidden magic or surprising behavior
- Better performance (less introspection)
- Easier to debug (clear execution flow)

## Recommendation

**Proceed with simplified design.** The current complexity is not justified by the features it provides. We can add back specific features if truly needed, but start simple.
