# Scorecard Widget

A Jupyter widget for displaying interactive policy evaluation scorecards. This widget integrates with the Metta
ecosystem to visualize training results and policy performance across different evaluations.

## Features

- Interactive scorecard visualization of policy evaluation results
- Click to select cells and view detailed metrics
- Open replay URLs directly in the browser
- Real-time data updates via Jupyter widget communication
- Configurable number of policies to display
- Multi-metric support with dynamic switching

## Installation

### Quick Installation (Recommended)

The easiest way to install the scorecard widget is using the Metta CLI:

```bash
metta install notebookwidgets
```

This command will automatically build the JavaScript components and install the Python package.

### Manual Installation

1. Install the Python package:

   ```bash
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

1. **Enable Development Mode**: Edit `scorecard_widget/ScorecardWidget.py` and set:

   ```python
   _DEV = True
   ```

2. **Start the Development Server**:

   ```bash
   pnpm run dev
   ```

   This starts the Vite development server on `http://localhost:5174`

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

2. **Disable Development Mode**: Edit `scorecard_widget/ScorecardWidget.py` and set:

   ```python
   _DEV = False
   ```

3. **Reinstall the Package** (if needed):
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```python
from scorecard_widget import ScorecardWidget

# Create a demo widget with sample data
widget = ScorecardWidget()
widget
```

### Loading Real Data

```python
from scorecard_widget import ScorecardWidget
from scorecard_widget.util import fetch_real_scorecard_data

# Fetch real data from the Metta backend
data = fetch_real_scorecard_data()
widget = ScorecardWidget()
widget.set_data(data)
widget
```

### Event Handling

```python
# Handle cell selection
def on_cell_selected(selected_cell):
    print(f"Selected: {selected_cell}")

widget.on_cell_selected(on_cell_selected)

# Handle metric changes
def on_metric_changed(metric):
    print(f"Metric changed to: {metric}")

widget.on_metric_changed(on_metric_changed)

# Handle replay opening
def on_replay_opened(replay_info):
    print(f"Opened replay: {replay_info}")

widget.on_replay_opened(on_replay_opened)
```

## Development Workflow

1. **Start Development**:
   - Set `_DEV = True` in `ScorecardWidget.py`
   - Run `pnpm run dev`
   - Start Jupyter and create/modify widgets

2. **Test Changes**:
   - Edit React/TypeScript files in `src/`
   - Changes will hot-reload in Jupyter
   - Check browser console for debugging

3. **Build for Production**:
   - Set `_DEV = False` in `ScorecardWidget.py`
   - Run `pnpm run build`
   - Test the built version

4. **Deploy**:
   - Commit your changes
   - Users can install with `metta install scorecardwidget`

## Important Notes

- **Always remember to toggle `_DEV` mode**: Set to `True` for development, `False` for production
- **Development server must be running**: When `_DEV = True`, ensure `pnpm run dev` is running
- **Build before distribution**: Run `pnpm run build` before setting `_DEV = False`
- **Installation builds automatically**: `metta install scorecardwidget` handles the build process

## Troubleshooting

### Widget Not Loading

- Check if `_DEV` mode matches your setup (dev server running vs. built assets)
- Verify the development server is running on port 5174 when `_DEV = True`
- Check browser console for JavaScript errors

### Development Server Issues

- Ensure port 5174 is available
- Check that `pnpm install` completed successfully
- Verify Vite configuration in `vite.config.js`

### Build Issues

- Run `pnpm install` to ensure dependencies are installed
- Check for TypeScript compilation errors during build
- Verify the built `index.js` file exists in `scorecard_widget/static/`
