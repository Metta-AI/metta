// From constants.hpp
const objectTypes: Record<number, [string, string]> = {
  0: ["agent", "A"],
  1: ["wall", "#"],
  2: ["mine", "g"],
  3: ["generator", "c"],
  4: ["altar", "a"],
  5: ["armory", "r"],
  6: ["lasery", "l"],
  7: ["lab", "b"],
  8: ["factory", "f"],
  9: ["temple", "t"],
  10: ["converter", "v"],
} as const;

// const objectsSchema = z.record(
//   z.string(),
//   z.object({
//     r: z.number(),
//     c: z.number(),
//     wall: z.number().optional(),
//     agent: z.number().optional(),
//     generator: z.number().optional(),
//     // TODO - more types
//   })
// );

export function objectsToMap(objects: unknown) {
  // fast parsing; zod is too slow
  if (!objects || typeof objects !== "object") {
    throw new Error("objects is not an object");
  }

  let width = 0,
    height = 0;
  const asciiObjects: { r: number; c: number; code: string }[] = [];

  for (const entry of Object.values(objects)) {
    if (!entry || typeof entry !== "object") {
      throw new Error(`objects entry ${entry} is not an object`);
    }
    const type_id = Number(entry["type"]);
    const r = Number(entry["r"]);
    const c = Number(entry["c"]);
    const ascii = objectTypes[type_id][1];
    width = Math.max(width, c);
    height = Math.max(height, r);
    asciiObjects.push({ r, c, code: ascii });
  }
  width += 1;
  height += 1;

  const map: string[][] = Array.from({ length: height }, () =>
    Array(width).fill(" ")
  );
  for (const object of asciiObjects) {
    map[object.r][object.c] = object.code;
  }

  return map.map((row) => row.join("")).join("\n");
}
