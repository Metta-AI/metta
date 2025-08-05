// vite.config.js
import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import react from '@vitejs/plugin-react';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

export default defineConfig({
  build: {
    outDir: "scorecard_widget/static",
    lib: {
      entry: ["src/index.tsx"],
      formats: ["es"],
    },
    rollupOptions: {
      external: [
        // Externalize the symlinked Scorecard component from Observatory
        './src/Scorecard.tsx',
      ],
    },
  },
  plugins: [
    react(),
    anywidget(),
    nodePolyfills({
      // More conservative polyfill configuration
      include: ['process'],
      globals: {
        global: true,
        process: true,
      },
    })
  ],
  resolve: {
    // Preserve symlinks to avoid conflicts
    preserveSymlinks: true,
  },
});
