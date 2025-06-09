/** A grid of booleans. */
export class Grid {
  private width: number;
  private height: number;
  private data: Uint8Array;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.data = new Uint8Array(width * height);
  }

  /** Set the value of a cell. */
  set(x: number, y: number, value: boolean) {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return;
    }
    this.data[y * this.width + x] = value ? 1 : 0;
  }

  /** Get the value of a cell. */
  get(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return false;
    }
    return this.data[y * this.width + x] === 1;
  }

  /** Clear the grid. */
  clear() {
    this.data.fill(0);
  }
}
