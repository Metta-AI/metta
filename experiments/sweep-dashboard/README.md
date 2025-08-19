# Sweep Analysis Dashboard

A modern React-based dashboard for analyzing hyperparameter sweeps with integrated Sky jobs monitoring, now with **real WandB data integration**.

## Features

- **Real Data**: Connects to WandB to fetch actual sweep data
- **Sky Jobs Monitor**: Real-time monitoring of Sky jobs with inline loading states
- **Interactive Visualizations**: Cost vs Score, Timeline, Distributions
- **Advanced Filtering**: Filter runs by score and cost ranges
- **Summary Metrics**: Key performance indicators at a glance
- **Responsive Design**: Works on desktop and mobile devices

## Getting Started

### Prerequisites

- Python 3.8+ (for backend)
- Node.js 18+ and npm (for frontend)
- WandB account and API access
- Sky CLI installed (for jobs monitoring)

### Quick Start

The easiest way to run both backend and frontend:

```bash
./run.sh
```

This will:
1. Install all dependencies (Python and Node)
2. Start the FastAPI backend on http://localhost:8000
3. Start the React frontend on http://localhost:3000
4. Handle graceful shutdown with Ctrl+C

### Manual Installation

If you prefer to run servers separately:

#### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export WANDB_ENTITY=your-entity
export WANDB_PROJECT=your-project

# Start backend server
python backend.py
```

#### Frontend Setup
```bash
# Install Node dependencies
npm install

# Start frontend server
npm run dev
```

## Architecture

### Component Structure

- `App.tsx` - Main application component
- `SweepDashboard.tsx` - Dashboard container
- `SkyJobsMonitor.tsx` - Sky jobs monitoring with clean loading states
- `SweepMetrics.tsx` - Summary statistics cards
- `SweepFilters.tsx` - Interactive filter controls
- `SweepCharts.tsx` - Visualization components using Chart.js
- `SweepSelector.tsx` - Sweep selection interface

### Loading States

The app implements clean, inline loading states without disruptive overlays:

- **Sky Jobs Monitor**: Shows an inline blue alert with spinner when refreshing
- **Data Loading**: Displays loading alerts while fetching sweep data
- **No Layout Disruption**: Loading indicators appear inline without covering content

### API Integration

Currently using mock data. To integrate with your backend:

1. Update `api/sweepApi.ts` to call your WandB API endpoints
2. Update `api/skyApi.ts` to call your Sky jobs backend
3. Configure the proxy in `vite.config.ts` to point to your API server

## Key Improvements Over Dash Version

1. **Better State Management**: React hooks provide clean, predictable state updates
2. **Simpler Loading States**: No complex callback chains - just conditional rendering
3. **Modern Development**: TypeScript, hot reload, better debugging
4. **Performance**: Virtual DOM ensures smooth updates
5. **Maintainability**: Component-based architecture is easier to extend

## Production Deployment

1. Build the app:
```bash
npm run build
```

2. Serve the `dist` folder with any static file server

## Environment Variables

Create a `.env` file for configuration:

```env
VITE_API_URL=http://localhost:8000
VITE_WANDB_ENTITY=your-entity
VITE_WANDB_PROJECT=your-project
```

## Contributing

The dashboard is designed to be easily extensible. To add new features:

1. Create new components in `src/components/`
2. Add API endpoints in `src/api/`
3. Update types in `src/types.ts`
4. Follow the existing patterns for loading states and error handling