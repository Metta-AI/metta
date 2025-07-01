# Metta Studio – Developer Guide

This directory contains the Next.js frontend for the Metta project that can be used locally. The Next.js app relies on `metta.studio.server` FastAPI server. A convenience launcher (`studio/start.py`) boots both servers in one command and opens the app in your browser.

## Prerequisites

1. **Node.js** – 22+
2. **pnpm** – run `corepack enable` once on your machine if you don't have it installed.

## First-time setup

```bash
# Install frontend deps
cd studio
pnpm install

# Generate the character-encoding lookup table used by the map viewer
pnpm run gen:encoding
```

`gen:encoding` calls a small Python snippet in the `mettagrid` package and writes the resulting JSON file to `src/lib/encoding.json`. Run it again whenever you change the encoder definitions.

## Running the app

From the repository root run:

```bash
./studio/start.py
```

The launcher will:

1. Build the frontend code if needed.
2. Start the **backend** (`metta.studio.server`).
3. Start the **Next.js** dev server via `pnpm start` (or `pnpm dev` if you use `--dev` flag).
4. Stream colored logs for both processes, prefixing them with `[BACKEND]` and `[FRONTEND]`.
5. Open `http://localhost:3000`.

Use `./studio/start.py --dev` to run the app in development mode. This will start the Next.js dev server that will automatically reload when you make changes to the code.

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal to shut everything down.

## Useful commands

```bash
pnpm dev                    # Next.js dev server only (if you want it without the backend)
uv run -m metta.studio.server  # Start the backend server only
pnpm run gen:encoding       # Regenerate encoding.json based on `metta.mettagrid.char_encoder` package.
```
