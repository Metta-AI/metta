#!/usr/bin/env tsx

import { execSync } from "child_process";
import { writeFileSync } from "fs";
import { resolve } from "path";
import { chdir } from "process";

async function main() {
  // Generate encoding.json
  const repoRootDir = resolve(__dirname, "../../");
  const mettagridDir = resolve(__dirname, "../../packages/mettagrid");
  const outputFile = resolve(__dirname, "../src/lib/encoding.json");

  chdir(mettagridDir);

  const pythonCmd = `uv run python -c 'import json; from mettagrid.mapgen.utils.ascii_grid import default_char_to_name; print(json.dumps(default_char_to_name()))'`;

  try {
    const output = execSync(pythonCmd, { encoding: "utf-8" });
    writeFileSync(outputFile, output);
    console.log(`Generated encoding.json at ${outputFile}`);
  } catch (error) {
    console.error("Error generating encoding:", error);
    process.exit(1);
  }
}

main();
