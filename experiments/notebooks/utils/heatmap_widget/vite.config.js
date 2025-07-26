// vite.config.js
import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import { nodePolyfills } from 'vite-plugin-node-polyfills';
import react from '@vitejs/plugin-react';

export default defineConfig({
  build: {
    outDir: "heatmap_widget/static",
    lib: {
      entry: ["src/index.tsx"],
      formats: ["es"],
    },
  },
  plugins: [
    react(),
    anywidget(),
    nodePolyfills({
      // Specifically include process polyfill
      include: ['process'],
      globals: {
        global: true,
        process: true,
      },
    })
  ],
});
