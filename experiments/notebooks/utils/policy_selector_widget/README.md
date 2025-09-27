# Policy Selector Widget

A marimo-compatible anywidget for selecting policies from the Metta AI system. This widget provides an interactive
interface for discovering, filtering, and selecting multiple policies with features like search, type filtering, tag
filtering, and multiselect functionality.

## Features

- **Search**: Filter policies by name with real-time text search
- **Type Filtering**: Filter by policy type (training_run or policy)
- **Tag Filtering**: Filter policies by their associated tags
- **Multiselect**: Select multiple policies with checkboxes
- **Configurable Display**: Show/hide metadata fields (tags, type, creation date)
- **Batch Operations**: Select all filtered policies or clear all selections
- **Callbacks**: React to selection and filter changes
- **Backend Integration**: Automatically fetch policies from the Metta backend API

## Installation

The widget is automatically built when you run `pnpm install` and `pnpm run build` from the root directory.

## Usage

### Basic Usage

```python
from policy_selector_widget import create_policy_selector_widget
from softmax.orchestrator.clients.scorecard_client import ScorecardClient

# Create the widget
policy_widget = create_policy_selector_widget()

# Load policies from backend
client = ScorecardClient()
policies_response = await client.get_policies()
policies = [
    {
        "id": p.id,
        "type": p.type,
        "name": p.name,
        "user_id": p.user_id,
        "created_at": p.created_at,
        "tags": p.tags,
    }
    for p in policies_response.policies
]

# Set policy data
policy_widget.set_policy_data(policies)

# Display the widget
policy_widget
```

### Configuration Options

```python
# Configure UI display options
policy_widget.set_ui_config(
    show_tags=True,
    show_type=True,
    show_created_at=True,
    max_displayed_policies=100,
)

# Set filters
policy_widget.set_search_term("navigation")
policy_widget.set_policy_type_filter(["training_run"])
policy_widget.set_tag_filter(["experimental"])
```

### Selection Management

```python
# Get selected policy IDs
selected_ids = policy_widget.get_selected_policies()

# Get full policy data for selected policies
selected_data = policy_widget.get_selected_policy_data()

# Programmatically select policies
policy_widget.select_policies(["policy-id-1", "policy-id-2"])

# Clear selection
policy_widget.clear_selection()
```

### Callbacks

```python
def on_selection_change(change_data):
    print(f"Selected {len(change_data['selected_policies'])} policies")

def on_filter_change(filter_data):
    print(f"Filter changed: {filter_data}")

policy_widget.on_selection_changed(on_selection_change)
policy_widget.on_filter_changed(on_filter_change)
```

## Example

See `experiments/marimo/policy-selector-example.py` for a complete working example that demonstrates all features of the
widget.

## API Reference

### PolicySelectorWidget

#### Methods

- `set_policy_data(policies: List[Dict])` - Set the list of policies to display
- `get_selected_policies() -> List[str]` - Get list of selected policy IDs
- `get_selected_policy_data() -> List[Dict]` - Get full data for selected policies
- `select_policies(policy_ids: List[str])` - Programmatically select policies
- `clear_selection()` - Clear all selections
- `set_search_term(term: str)` - Set search filter
- `set_policy_type_filter(types: List[str])` - Set policy type filter
- `set_tag_filter(tags: List[str])` - Set tag filter
- `set_ui_config(**config)` - Configure UI display options
- `on_selection_changed(callback)` - Register selection change callback
- `on_filter_changed(callback)` - Register filter change callback

#### Policy Data Format

Each policy should be a dictionary with these fields:

```python
{
    "id": str,              # Unique policy ID
    "type": str,            # "training_run" or "policy"
    "name": str,            # Display name
    "user_id": str,         # Optional user ID
    "created_at": str,      # ISO timestamp
    "tags": List[str],      # List of tags
}
```

## Development

### Building

```bash
# Install dependencies
pnpm install

# Build the widget
pnpm run build
```

### Project Structure

```
policy_selector_widget/
├── src/                          # React TypeScript components
│   ├── index.tsx                 # Main widget entry point
│   ├── PolicySelector.tsx        # Main component
│   ├── types.ts                  # TypeScript types
│   └── styles.css               # CSS styles
├── policy_selector_widget/       # Python package
│   ├── __init__.py
│   ├── PolicySelectorWidget.py   # Main widget class
│   ├── util.py                  # Utility functions
│   └── static/                  # Built assets
├── package.json                 # NPM configuration
├── vite.config.js               # Build configuration
├── tsconfig.json                # TypeScript configuration
└── turbo.jsonc                  # Turbo configuration
```
