// vite.config.js
import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";

export default defineConfig({
  build: {
    outDir: "heatmap_widget/static",
    lib: {
      entry: ["src/index.tsx"],
      formats: ["es"],
    },
  },
  plugins: [anywidget()],
  define: {
    // Polyfill process for browser environment (needed by plotly.js)
    'process.env': {},
    'process.version': '"v16.0.0"',
    'process.platform': '"browser"',
    'process.browser': 'true',
    'global': 'globalThis',
  },
});
