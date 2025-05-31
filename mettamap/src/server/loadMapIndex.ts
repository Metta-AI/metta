import fs from "fs/promises";
import path from "path";

import { MAP_INDEX_FILE } from "./constants";

export type MapIndex = Record<string, Record<string, string[]>>;

export async function loadMapIndex(): Promise<MapIndex> {
  try {
    const indexFilePath = path.resolve(process.cwd(), MAP_INDEX_FILE);
    const indexContent = await fs.readFile(indexFilePath, "utf-8");
    return JSON.parse(indexContent) as MapIndex;
  } catch (error) {
    throw new Error(
      `Failed to load map index: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  }
}
