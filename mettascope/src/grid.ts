export class Grid {
  private width: number;
  private height: number;
  private data: Uint8Array;  // or Uint32Array if you want to pack 32 bools per int

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.data = new Uint8Array(width * height);
  }

  // Fast index calculation - no string creation or hash lookups
  set(x: number, y: number, value: boolean) {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return;
    }
    this.data[y * this.width + x] = value ? 1 : 0;
  }

  get(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return false;
    }
    return this.data[y * this.width + x] === 1;
  }

  clear() {
    this.data.fill(0);
  }
}
