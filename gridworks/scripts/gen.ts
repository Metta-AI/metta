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

  const pythonCmd = `uv run python -c 'import json; from mettagrid.util.char_encoder import NAME_TO_CHAR; print(json.dumps(NAME_TO_CHAR))'`;

  try {
    const output = execSync(pythonCmd, { encoding: "utf-8" });
    writeFileSync(outputFile, output);
    console.log(`Generated encoding.json at ${outputFile}`);
  } catch (error) {
    console.error("Error generating encoding:", error);
    process.exit(1);
  }

  // Generate Pydantic schemas
  chdir(repoRootDir);
  const output = execSync("uv run python metta/gridworks/generate_schemas.py", {
    encoding: "utf-8",
  });
  writeFileSync(resolve(__dirname, `../src/lib/schemas.json`), output);
}

main();
