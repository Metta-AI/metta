import defaultEncoding from "@/lib/encoding.json" assert { type: "json" };

import { StorableMap } from "./api";

export class AsciiEncoding {
  charToName: Record<string, string>;
  nameToChar: Record<string, string>;

  constructor(charToName: Record<string, string>) {
    this.charToName = charToName;
    this.nameToChar = Object.fromEntries(
      Object.entries(charToName).map(([char, name]) => [name, char])
    );
  }

  getNameFromChar(char: string): string {
    return this.charToName[char] ?? "unknown";
  }

  getCharFromName(name: string): string {
    return this.nameToChar[name] ?? "?";
  }

  static default(): AsciiEncoding {
    return new AsciiEncoding(defaultEncoding);
  }
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

  static fromAscii(data: string, encoding: AsciiEncoding): MettaGrid {
    // Parse the data
    const lines = data.trim().split("\n");
    const width = Math.max(...lines.map((line) => line.length));
    const height = lines.length;
    const objects: MettaObject[] = [];
    lines.forEach((line, y) => {
      line.split("").forEach((char, x) => {
        const object = MettaObject.fromObjectName(
          y,
          x,
          encoding.getNameFromChar(char)
        );
        if (object) {
          objects.push(object);
        }
      });
    });
    return new MettaGrid({ width, height, objects });
  }

  static fromStorableMap(map: StorableMap) {
    return this.fromAscii(
      map.data,
      new AsciiEncoding(map.frontmatter.char_to_name)
    );
  }

  toAscii(encoding: AsciiEncoding): string {
    const emptyAscii = encoding.getCharFromName("empty");
    return this.grid
      .map((row) =>
        row
          .map((cell) =>
            cell ? encoding.getCharFromName(cell.name) : emptyAscii
          )
          .join("")
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
