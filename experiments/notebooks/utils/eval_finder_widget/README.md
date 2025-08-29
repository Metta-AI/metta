# Eval Finder Widget

A Jupyter widget for discovering and selecting evaluations in the Metta AI ecosystem. This widget helps you explore
available evaluations, filter them by compatibility and difficulty, and understand prerequisite relationships.

## Features

- **🌳 Multiple View Modes**: Tree, list, and category-grouped views
- **🔍 Smart Search**: Filter by name, description, or tags
- **📊 Category-based Organization**: Evaluations organized by category (navigation, memory, arena, etc.)
- **🎯 Multi-select**: Choose multiple evaluations with checkboxes
- **⚡ Real-time Filtering**: Instant updates as you change filters
- **📈 Policy-aware**: Shows which evaluations have been completed for your policies

## Installation

```sh
metta install notebookwidgets
```

### Prerequisites

Make sure you're in the Metta repository root directory and have the proper Python environment activated (with
`metta.app_backend` available).

### Quick Installation (Recommended)

Build the widget assets:

```bash
cd experiments/notebooks/utils/eval_finder_widget
pnpm install
pnpm run build
```

The widget is now ready to use in your notebooks.

### Manual Installation

1. Install the Python package:

   ```bash
   cd experiments/notebooks/utils/eval_finder_widget
   pip install -e .
   ```

2. Build the JavaScript components:
   ```bash
   pnpm install
   pnpm run build
   ```

## Development

### Development Mode Setup

For active development, you can use the development mode which provides hot-reloading:

1. **Enable Development Mode**: Edit `eval_finder_widget/EvalFinderWidget.py` and set:

   ```python
   _DEV = True
   ```

2. **Start the Development Server**:

   ```bash
   pnpm run dev
   ```

   This starts the Vite development server on `http://localhost:5175`

3. **Develop with Hot Reload**:
   - The widget will now load JavaScript from the development server
   - Changes to React/TypeScript files will automatically reload
   - View console output for debugging

### Production Mode

For production use or when you're done developing:

1. **Build the Production Assets**:

   ```bash
   pnpm run build
   ```

2. **Disable Development Mode**: Edit `eval_finder_widget/EvalFinderWidget.py` and set:
   ```python
   _DEV = False
   ```

## Usage

### Basic Usage

```python
from experiments.notebooks.utils.eval_finder_widget.eval_finder_widget import EvalFinderWidget
from experiments.notebooks.utils.eval_finder_widget.eval_finder_widget.util import create_demo_eval_finder_widget

# Create a demo widget with sample data
widget = create_demo_eval_finder_widget()
widget
```

### Policy-Aware Real Data

```python
from experiments.notebooks.utils.eval_finder_widget.eval_finder_widget import EvalFinderWidget
from experiments.notebooks.utils.eval_finder_widget.eval_finder_widget.util import fetch_eval_data_for_policies

# Create widget and fetch real evaluation data for specific policies
widget = EvalFinderWidget()

# Fetch evaluations contextually based on policies/runs
eval_data = fetch_eval_data_for_policies(
    training_run_ids=["user.navigation_run"],  # Training runs
    run_free_policy_ids=["policy-abc-123"],    # Standalone policies
    category_filter=["navigation", "memory"]    # Optional: filter by categories
)

# Set the data on the widget
widget.set_eval_data(
    evaluations=eval_data["evaluations"],
    categories=eval_data["categories"],
)

widget
```

### Event Handling

```python
# Handle selection changes
def on_selection_changed(event):
    selected_evals = event.get('selected_evals', [])
    print(f"Selected {len(selected_evals)} evaluations")

widget.on_selection_changed(on_selection_changed)

# Handle filter changes
def on_filter_changed(event):
    print(f"Filters updated: {event}")

widget.on_filter_changed(on_filter_changed)
```

### Programmatic Control

```python
# Set filters programmatically
widget.set_category_filter(["navigation"])

# Control selection
widget.select_evals(["navigation/labyrinth", "memory/hard"])
widget.clear_selection()

# Change view mode
widget.set_view_mode("tree")  # "tree", "list", "category"

# Search
widget.set_search_term("memory")
```

## Integration with Scorecard Widget

The eval finder integrates seamlessly with the existing scorecard widget:

```python
from eval_finder_widget import EvalFinderWidget
from scorecard_widget import ScorecardWidget

# Select evaluations with the finder
finder = EvalFinderWidget()
# ... user selects evaluations ...

# Use selected evals in scorecard
selected_evals = finder.get_selected_evals()
scorecard = ScorecardWidget()
# Generate scorecard with selected evaluations
# ... scorecard setup ...
```

## Marimo Integration

See `experiments/marimo/eval-finder-example.py` for a complete example of using the widget in Marimo notebooks.

## Data Sources

The widget leverages existing Metta infrastructure:

- **Evaluation Discovery**: Uses `ScorecardClient` to fetch eval names for policies
- **Policy-aware**: Shows which evaluations have been completed and their performance
- **No New Backend**: Reuses existing scorecard backend infrastructure

## Architecture

The widget follows the same pattern as the existing scorecard widget:

- **Python Backend**: `EvalFinderWidget` class with traitlets for data sync
- **React Frontend**: TypeScript components for UI rendering
- **anywidget Framework**: Seamless Jupyter integration
- **Vite Build System**: Modern frontend tooling

## File Structure

```
eval_finder_widget/
├── README.md
├── package.json             # dependencies and scripts
├── vite.config.js           # Vite build configuration
├── should_build.sh          # Build script
├── eval_finder_widget/      # Python package
│   ├── __init__.py
│   ├── EvalFinderWidget.py  # Main widget class
│   ├── util.py              # Utility functions
│   └── static/              # Built JavaScript assets
└── src/                     # TypeScript/React source
    ├── index.tsx            # Entry point
    ├── EvalFinder.tsx       # Main component
    ├── EvalTreeView.tsx     # Tree view component
    ├── EvalListView.tsx     # List view component
    ├── FilterPanel.tsx      # Filter controls
    ├── SearchBar.tsx        # Search component
    ├── styles.css           # Widget styles
    └── types.ts             # TypeScript types
```

## Development Workflow

1. **Start Development**:
   - Set `_DEV = True` in `EvalFinderWidget.py`
   - Run `pnpm run dev`
   - Start Jupyter and create/modify widgets

2. **Test Changes**:
   - Edit React/TypeScript files in `src/`
   - Changes will hot-reload in Jupyter
   - Check browser console for debugging

3. **Build for Production**:
   - Set `_DEV = False` in `EvalFinderWidget.py`
   - Run `pnpm run build`
   - Test the built version

4. **Deploy**:
   - Commit your changes
   - Users can install with `metta install evalfinderwidget`

## Important Notes

- **Always remember to toggle `_DEV` mode**: Set to `True` for development, `False` for production
- **Development server must be running**: When `_DEV = True`, ensure `pnpm run dev` is running
- **Build before distribution**: Run `pnpm run build` before setting `_DEV = False`
- **Uses existing backend**: No new API endpoints required

## Troubleshooting

### Widget Not Loading

- Check if `_DEV` mode matches your setup (dev server running vs. built assets)
- Verify the development server is running on port 5175 when `_DEV = True`
- Check browser console for JavaScript errors

### Development Server Issues

- Ensure port 5175 is available
- Check that `pnpm install` completed successfully
- Verify Vite configuration in `vite.config.js`

### Build Issues

- Run `pnpm install` to ensure dependencies are installed
- Check for TypeScript compilation errors during build
- Verify the built `index.js` file exists in `eval_finder_widget/static/`
