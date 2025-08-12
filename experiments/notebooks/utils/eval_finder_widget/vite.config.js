import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { nodePolyfills } from "vite-plugin-node-polyfills";
import anywidget from "@anywidget/vite";

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills(),
    anywidget()
  ],
  build: {
    outDir: "eval_finder_widget/static",
    lib: {
      entry: "src/index.tsx",
      formats: ["es"],
      fileName: "index"
    },
    rollupOptions: {
      external: [],
    }
  },
});