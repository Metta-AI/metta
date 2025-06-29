import encoding from "./encoding.json" with { type: "json" };

const typedEncoding: Record<string, string[]> = encoding;

const asciiToNameCache = new Map<string, string>();
function asciiToName(ascii: string): string {
  if (asciiToNameCache.has(ascii)) {
    return asciiToNameCache.get(ascii)!;
  }
  for (const [name, chars] of Object.entries(typedEncoding)) {
    if (chars.includes(ascii)) {
      asciiToNameCache.set(ascii, name);
      return name;
    }
  }
  throw new Error(`Invalid character: '${ascii}'`);
}

function nameToAscii(name: string): string {
  if (!typedEncoding[name]) {
    throw new Error(`Invalid object name: '${name}'`);
  }
  return typedEncoding[name][0]!;
}

export type Cell = {
  // these are intentionally `r` and `c` instead of `x` and `y` so that we don't confuse them with screen coordinates
  r: number;
  c: number;
};

export class MettaObject {
  readonly name: string;
  readonly r: number;
  readonly c: number;

  private constructor(data: { name: string; r: number; c: number }) {
    this.name = data.name;
    this.r = data.r;
    this.c = data.c;
  }

  static fromAscii(
    r: number,
    c: number,
    ascii: string
  ): MettaObject | undefined {
    const objectName = asciiToName(ascii);
    return MettaObject.fromObjectName(r, c, objectName);
  }

  static fromObjectName(
    r: number,
    c: number,
    name: string
  ): MettaObject | undefined {
    if (name === "empty") {
      return undefined;
    }
    return new MettaObject({ name, r, c });
  }

  get ascii(): string {
    return nameToAscii(this.name);
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

  static empty(width: number, height: number): MettaGrid {
    return new MettaGrid({
      width,
      height,
      objects: Array.from({ length: width * height }, (_, i) => {
        const r = Math.floor(i / width);
        const c = i % width;
        const name =
          r === 0 || r === height - 1 || c === 0 || c === width - 1
            ? "wall"
            : "empty";
        return MettaObject.fromObjectName(r, c, name);
      }).filter((o) => o !== undefined),
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
    const emptyAscii = nameToAscii("empty");
    return this.grid
      .map((row) =>
        row.map((cell) => (cell ? cell.ascii : emptyAscii)).join("")
      )
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

  replaceCellByName(r: number, c: number, name: string): MettaGrid {
    const newObject = MettaObject.fromObjectName(r, c, name);
    let replaced = false;
    const newObjects = this.objects
      .map((o) => {
        if (o.r === r && o.c === c) {
          replaced = true;
          return newObject;
        } else {
          return o;
        }
      })
      .filter((o) => o !== undefined);

    if (!replaced && newObject) {
      newObjects.push(newObject);
    }

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
