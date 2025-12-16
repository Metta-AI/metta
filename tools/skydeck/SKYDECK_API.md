# Skydeck API

This module provides an API for external tools (like Skydeck) to discover and introspect metta training tools.

## Usage

```bash
uv run ./tools/skydeck/api.py <command> [args]
```

## Commands

### `tools` - List all available tools

Discovers all tool functions (`train`, `evaluate`, `play`, `replay`) in the `recipes.experiment` and `recipes.prod`
modules.

```bash
uv run ./tools/skydeck/api.py tools
```

**Output:**

```json
{
  "tools": {
    "recipes.experiment.arena.train": "TrainTool",
    "recipes.experiment.arena.evaluate": "EvaluateTool",
    "recipes.experiment.arena.play": "PlayTool",
    "recipes.experiment.cog_arena.train": "TrainTool",
    "recipes.experiment.scratchpad.daveey.train": "TrainTool",
    ...
  }
}
```

### `schema` - Get Pydantic schema for a tool

Extracts the full Pydantic schema for a tool, including all nested configuration fields.

```bash
uv run ./tools/skydeck/api.py schema <import_path>
```

**Examples:**

```bash
# Get schema for a recipe tool
uv run ./tools/skydeck/api.py schema arena.train

# Get schema for a direct config class
uv run ./tools/skydeck/api.py schema metta.rl.trainer_config.TrainerConfig
```

**Output (success):**

```json
{
  "schema": {
    "run": {
      "type": "str",
      "default": null,
      "required": false
    },
    "trainer.total_timesteps": {
      "type": "int",
      "default": 50000000000,
      "required": false
    },
    ...
  }
}
```

**Output (failure):**

```json
{
  "invalid_keys": ["invalid.path"]
}
```

## Schema Format

Each field in the schema contains:

| Key           | Type    | Description                                                                       |
| ------------- | ------- | --------------------------------------------------------------------------------- |
| `type`        | string  | Human-readable type (e.g., `int`, `str`, `bool`, `float \| None`, `Literal[...]`) |
| `default`     | any     | Default value, or `null` if none                                                  |
| `required`    | boolean | Whether the field is required                                                     |
| `description` | string  | (Optional) Field description from Pydantic                                        |

## Integration

This API is designed to be called by external tools. All output is JSON to stdout. Errors are written to stderr but the
exit code is always 0 to allow partial results.
