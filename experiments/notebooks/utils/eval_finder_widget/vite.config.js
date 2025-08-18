// vite.config.js
import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import react from "@vitejs/plugin-react";
import { nodePolyfills } from "vite-plugin-node-polyfills";

export default defineConfig({
  build: {
    outDir: "eval_finder_widget/static",
    lib: {
      entry: ["src/index.tsx"],
      formats: ["es"],
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
});
