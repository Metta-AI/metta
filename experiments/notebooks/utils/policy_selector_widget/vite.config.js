// vite.config.js
import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import react from "@vitejs/plugin-react";
import { nodePolyfills } from "vite-plugin-node-polyfills";

export default defineConfig({
  build: {
    outDir: "policy_selector_widget/static",
    lib: {
      entry: ["src/index.tsx"],
      formats: ["es"],
    },
    rollupOptions: {
      external: (id) => {
        // Don't externalize react and react-dom for anywidget builds
        return false;
      },
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