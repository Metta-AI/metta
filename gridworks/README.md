# Gridworks

This directory contains the web UI for viewing and editing the environments and maps.

The Next.js app relies on the `metta.gridworks.server` FastAPI server. A convenience launcher (`gridworks/start.py`) boots both servers in one command and opens the app in your browser.

The project is designed to be run locally.

## Running the app

From the repository root run:

```bash
./gridworks/start.py
```

The launcher will:

1. Build the frontend code if needed.
2. Start the **backend** (`metta.gridworks.server`).
3. Start the **Next.js** dev server via `pnpm start` (or `pnpm dev` if you use `--dev` flag).
4. Stream colored logs for both processes, prefixing them with `[BACKEND]` and `[FRONTEND]`.
5. Open `http://localhost:3000`.

Use `./gridworks/start.py --dev` to run the app in development mode. This will start the Next.js dev server that will automatically reload when you make changes to the code.

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal to shut everything down.

## Useful commands

```bash
pnpm dev                    # Next.js dev server only (if you want it without the backend)
uv run -m metta.gridworks.server  # Start the backend server only
pnpm run gen:encoding       # Regenerate encoding.json based on `mettagrid.util.char_encoder` package.
```

## Generating encodings

```bash
# Generate the character-encoding lookup table used by the map viewer
pnpm run gen:encoding
```

`gen:encoding` calls a small Python snippet in the `mettagrid` package and writes the resulting JSON file to `src/lib/encoding.json`. Run it again whenever you change the encoder definitions.
