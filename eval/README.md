# Policy Evaluation Dashboard

A web application for visualizing policy evaluation results using DuckDB-WASM. This app runs entirely in the browser, with no backend server required.

## Features

- Loads DuckDB files directly in the browser
- Interactive heatmap visualization using Plotly.js
- Pure frontend solution - no backend required
- TypeScript for type safety
- Modern React with hooks

## Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file with your database URI:
   ```
   VITE_EVAL_DB_URI=path/to/your/eval.db
   ```
   The URI can be:
   - A local file path (e.g., `file:///path/to/eval.db`)
   - A remote URL (e.g., `https://example.com/eval.db`)

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser to the URL shown in the terminal (typically http://localhost:5173)

## Production Build

1. Build the app:
   ```bash
   npm run build
   ```

2. Preview the production build:
   ```bash
   npm run preview
   ```

## How It Works

1. The app loads the DuckDB file specified in `VITE_EVAL_DB_URI`
2. It runs SQL queries directly in the browser using DuckDB-WASM
3. The results are rendered as an interactive heatmap using Plotly.js
4. All data processing happens in the browser - no server required

## Environment Variables

- `VITE_EVAL_DB_URI`: Path to your DuckDB file
  - Local file: `file:///absolute/path/to/eval.db`
  - Remote file: `https://example.com/path/to/eval.db` 