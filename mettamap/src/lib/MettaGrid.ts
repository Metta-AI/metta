const objectTypes = [
  ["agent", "A"], // 0
  ["wall", "#"], // 1
  ["mine", "g"], // 2
  ["generator", "c"], // 3
  ["altar", "a"], // 4
  ["armory", "r"], // 5
  ["lasery", "l"], // 6
  ["lab", "b"], // 7
  ["factory", "f"], // 8
  ["temple", "t"], // 9
  ["converter", "v"], // 10
  // TODO - empty?
] as const satisfies [string, string][];

export type ObjectName = (typeof objectTypes)[number][0] | "empty";

const asciiToTypeId = Object.fromEntries(
  objectTypes.map(([, ascii], i) => [ascii, i])
);

const objectNameToTypeId: Record<ObjectName, number> = Object.fromEntries(
  objectTypes.map(
    ([name]) =>
      [name, objectTypes.findIndex(([n]) => n === name)] satisfies [
        ObjectName,
        number,
      ]
  )
) as Record<ObjectName, number>;

export type ItemObjectName = Exclude<ObjectName, "wall" | "agent" | "empty">;

export type Cell = {
  // these are intentionally `r` and `c` instead of `x` and `y` so that we don't confuse them with screen coordinates
  r: number;
  c: number;
};

export class MettaObject {
  readonly type: number;
  readonly r: number;
  readonly c: number;

  constructor(data: { type: number; r: number; c: number }) {
    this.type = data.type;
    this.r = data.r;
    this.c = data.c;
  }

  static fromAscii(
    r: number,
    c: number,
    ascii: string
  ): MettaObject | undefined {
    if (ascii === "." || ascii === " ") {
      return undefined;
    }
    const objectType = asciiToTypeId[ascii];
    if (objectType === undefined) {
      throw new Error(`Invalid character: '${ascii}' at ${r},${c}`);
    }
    return new MettaObject({ type: objectType, r, c });
  }

  static fromObjectName(
    r: number,
    c: number,
    name: ObjectName
  ): MettaObject | undefined {
    if (name === "empty") {
      return undefined;
    }
    const typeId = objectNameToTypeId[name];
    if (typeId === undefined) {
      throw new Error(`Invalid object name: '${name}' at ${r},${c}`);
    }
    return new MettaObject({
      type: typeId,
      r,
      c,
    });
  }

  get name(): ObjectName {
    return objectTypes[this.type][0];
  }

  get ascii(): string {
    return objectTypes[this.type][1];
  }
}

export class MettaGrid {
  private readonly grid: (MettaObject | null)[][];

  constructor(
    private readonly data: {
      width: number;
      height: number;
      objects: MettaObject[];
    }
  ) {
    this.grid = new Array(data.height)
      .fill(null)
      .map(() => new Array(data.width).fill(null));

    for (const object of data.objects) {
      this.grid[object.r][object.c] = object;
    }
  }

  static empty(width: number, height: number, borderWidth = 1): MettaGrid {
    return new MettaGrid({
      width,
      height,
      objects: Array.from({ length: width * height }, (_, i) => {
        const r = Math.floor(i / width);
        const c = i % width;
        return new MettaObject({
          type: 1,
          r,
          c,
        });
      }),
    });
  }

  static fromAscii(asciiMap: string) {
    // Parse the data
    const lines = asciiMap.trim().split("\n");
    const width = Math.max(...lines.map((line) => line.length));
    const height = lines.length;
    const objects: MettaObject[] = [];
    lines.forEach((line, y) => {
      line.split("").forEach((char, x) => {
        const object = MettaObject.fromAscii(y, x, char);
        if (object) {
          objects.push(object);
        }
      });
    });
    return new MettaGrid({ width, height, objects });
  }

  toAscii(): string {
    return this.grid
      .map((row) => row.map((cell) => cell?.ascii ?? ".").join(""))
      .join("\n");
  }

  object(cell: Cell): MettaObject | null {
    return this.grid[cell.r][cell.c];
  }

  get width(): number {
    return this.data.width;
  }

  get height(): number {
    return this.data.height;
  }

  get objects(): MettaObject[] {
    return this.data.objects;
  }

  replaceCellByName(r: number, c: number, name: ObjectName) {
    const newObjects = this.objects
      .map((o) => {
        if (o.r === r && o.c === c) {
          return MettaObject.fromObjectName(r, c, name);
        } else {
          return o;
        }
      })
      .filter((o) => o !== undefined);

    return new MettaGrid({
      ...this.data,
      objects: newObjects,
    });
  }

  cellInGrid(cell: Cell): boolean {
    return (
      cell.r >= 0 && cell.r < this.height && cell.c >= 0 && cell.c < this.width
    );
  }
}
