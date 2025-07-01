# Observatory FE

This is the FE for the app deployed at https://observatory.softmax-research.net/
Talks to the server in `/app_backend` - see instructions on how to run it in `/app_backend/README.md`

## Setup

Ensure Observatory is installed through the Metta setup tool:
```bash
# From the metta root directory
./metta.sh install observatory-fe
```

## Development

1. If you need to manually install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to the URL shown in the terminal (typically http://localhost:5173)

## Production Build

1. Build the app:
   ```bash
   npm run build
   ```
