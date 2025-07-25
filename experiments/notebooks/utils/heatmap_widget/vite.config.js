// vite.config.js
import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";

export default defineConfig({
  build: {
    outDir: "heatmap_widget/static",
    lib: {
      entry: ["src/index.js"],
      formats: ["es"],
    },
  },
  plugins: [anywidget()],
});
