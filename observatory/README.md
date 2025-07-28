# Observatory FE

This is the FE for the app deployed at https://observatory.softmax-research.net/
Talks to the server in `/app_backend` - see instructions on how to run it in `/app_backend/README.md`

## Setup

Ensure Observatory is installed through the Metta setup tool:
```bash
# From the metta root directory
metta install nodejs
```

## Development

1. If you need to manually install dependencies:
   ```bash
   pnpm install
   ```

2. Start the development server:
   ```bash
   pnpm run dev
   ```
   or
   ```bash
   metta local observatory [--backend {prod, local}]
   ```

3. Open your browser to the URL shown in the terminal (typically http://localhost:5173)


4. For linting and formatting, run `pnpm run format ` and `pnpm run check:fix`

## Production Build

1. Build the app:
   ```bash
   pnpm run build
   ```
