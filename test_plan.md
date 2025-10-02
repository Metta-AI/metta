# Tool Registry Refactoring - Test Plan

This test plan verifies the Recipe abstraction and tool discovery improvements.

## 1. List All Recipes

```bash
./tools/run.py --list
```

**Expected output:**
- Should show "Available Recipes:" header
- Should list all recipes under `experiments.recipes` (e.g., `arena`, `navigation`, `ci`, etc.)
- Each recipe should show its available tools indented beneath it
- Format: recipe name, then tools with `└─ tool_name`

**Success criteria:** Multiple recipes listed with their tools clearly organized

## 2. List Tools for Specific Recipe

```bash
./tools/run.py arena --list
```

**Expected output:**
- Should show "Available tools in experiments.recipes.arena:"
- Should list tools like `train`, `evaluate`, `play`, `replay`
- Should NOT show "Available 'arena' implementations:" (that would be bare tool behavior)

**Success criteria:** Shows tools for arena recipe only, not all recipes

## 3. Short Form Recipe Listing

```bash
./tools/run.py navigation --list
```

**Expected output:**
- Should show "Available tools in experiments.recipes.navigation:"
- Should include both inferred tools (train, evaluate, play, replay) and any explicit tools

**Success criteria:** Recipe path is resolved correctly without full `experiments.recipes.` prefix

## 4. Tool Resolution (Two-Token Form)

```bash
./tools/run.py train arena --dry-run
```

**Expected output:**
- Should resolve to `experiments.recipes.arena.train`
- Should show dry-run success message (not error)

**Success criteria:** Resolves `train arena` → `arena.train` correctly

## 5. Tool Resolution (Dotted Form)

```bash
./tools/run.py arena.train --dry-run
```

**Expected output:**
- Should resolve to `experiments.recipes.arena.train`
- Should show dry-run success message

**Success criteria:** Standard dotted notation still works

## 6. Tool Alias Resolution

```bash
./tools/run.py arena.eval --dry-run
```

**Expected output:**
- Should resolve `eval` alias to `evaluate` tool
- Should show dry-run success message

**Success criteria:** Alias `eval` → `evaluate` works correctly

## 7. Inferred Tool Works

```bash
./tools/run.py arena.play --dry-run
```

**Expected output:**
- Should successfully infer PlayTool from arena recipe's configs
- Should show dry-run success message

**Success criteria:** Tool inference from recipe configs works

## 8. Error Handling - Unknown Tool

```bash
./tools/run.py arena.nonexistent
```

**Expected output:**
- Should show "Error: Could not find tool 'arena.nonexistent'"
- Should show helpful hint about available inferred tools
- Should exit with non-zero code

**Success criteria:** Clear error message with helpful suggestions

## 9. Error Handling - Unknown Recipe

```bash
./tools/run.py fake_recipe.train
```

**Expected output:**
- Should show "Error: Could not find tool 'fake_recipe.train'"
- Should exit with non-zero code

**Success criteria:** Fails gracefully with clear error

## 10. Run Full Test Suite

```bash
uv run pytest common/tests/tool/ -v
```

**Expected output:**
- All tests should pass
- Should see ~23 tests total (15 in test_run_tool.py + 8 in test_arg_parsing.py)

**Success criteria:** All tests pass with no failures

## Summary

If all 10 commands produce the expected output, the refactoring is working correctly:
- ✅ Recipe discovery and listing works
- ✅ Tool resolution (both forms) works
- ✅ Tool inference from recipes works
- ✅ Alias resolution works
- ✅ Error handling is helpful
- ✅ All tests pass
