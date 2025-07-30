/**
 * Vec2f class - Represents a 2D vector with x and y components.
 */
class Vec2f {
  public data: Float32Array

  constructor(x: number, y: number) {
    // Use Float32Array as backing storage for better performance
    this.data = new Float32Array(2)
    this.data[0] = x
    this.data[1] = y
  }

  x(): number {
    return this.data[0]
  }
  y(): number {
    return this.data[1]
  }
  setX(x: number) {
    this.data[0] = x
  }
  setY(y: number) {
    this.data[1] = y
  }

  /** Returns a new vector that is the sum of this vector and v. */
  add(v: Vec2f): Vec2f {
    return new Vec2f(this.x() + v.x(), this.y() + v.y())
  }

  /** Returns a new vector that is this vector minus v. */
  sub(v: Vec2f): Vec2f {
    return new Vec2f(this.x() - v.x(), this.y() - v.y())
  }

  /** Returns a new vector that is this vector multiplied by scalar s. */
  mul(s: number): Vec2f {
    return new Vec2f(this.x() * s, this.y() * s)
  }

  /** Returns a new vector that is this vector divided by scalar s. */
  div(s: number): Vec2f {
    return new Vec2f(this.x() / s, this.y() / s)
  }

  /** Returns the length (magnitude) of this vector. */
  length(): number {
    return Math.sqrt(this.x() * this.x() + this.y() * this.y())
  }

  /** Returns a new vector that is this vector normalized (length = 1). */
  normalize(): Vec2f {
    const length = this.length()
    return new Vec2f(this.x() / length, this.y() / length)
  }
}

/**
 * Mat3f class - Represents a 3x3 matrix for 2D transformations.
 * Matrix is organized in row-major order:
 * | 0 1 2 |
 * | 3 4 5 |
 * | 6 7 8 |
 */
class Mat3f {
  public data: Float32Array

  constructor(a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) {
    // Use Float32Array as backing storage for better performance
    this.data = new Float32Array(9)
    this.data[0] = a
    this.data[1] = b
    this.data[2] = c
    this.data[3] = d
    this.data[4] = e
    this.data[5] = f
    this.data[6] = g
    this.data[7] = h
    this.data[8] = i
  }

  /** Row-column access - get element at row r, column c (both 0-based) */
  get(r: number, c: number): number {
    return this.data[r * 3 + c]
  }

  /** Set element at row r, column c (both 0-based) to value */
  set(r: number, c: number, value: number): this {
    this.data[r * 3 + c] = value
    return this
  }

  /** Returns an identity matrix (no transformation). */
  static identity(): Mat3f {
    return new Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1)
  }

  /** Returns a translation matrix for moving by (x, y). */
  static translate(x: number, y: number): Mat3f {
    return new Mat3f(1, 0, x, 0, 1, y, 0, 0, 1)
  }

  /** Returns a scaling matrix for scaling by factors (x, y). */
  static scale(x: number, y: number): Mat3f {
    return new Mat3f(x, 0, 0, 0, y, 0, 0, 0, 1)
  }

  /** Returns a rotation matrix for rotating by angle (in radians). */
  static rotate(angle: number): Mat3f {
    const c = Math.cos(angle)
    const s = Math.sin(angle)
    return new Mat3f(c, s, 0, -s, c, 0, 0, 0, 1)
  }

  /**
   * Transforms a vector v using this matrix and returns the result.
   * Converts the 2D vector to homogeneous coordinates for the transformation.
   */
  transform(v: Vec2f): Vec2f {
    return new Vec2f(
      this.get(0, 0) * v.x() + this.get(0, 1) * v.y() + this.get(0, 2),
      this.get(1, 0) * v.x() + this.get(1, 1) * v.y() + this.get(1, 2)
    )
  }

  /** Returns a new matrix that is the result of multiplying this matrix by m. */
  mul(m: Mat3f): Mat3f {
    return new Mat3f(
      this.get(0, 0) * m.get(0, 0) + this.get(0, 1) * m.get(1, 0) + this.get(0, 2) * m.get(2, 0),
      this.get(0, 0) * m.get(0, 1) + this.get(0, 1) * m.get(1, 1) + this.get(0, 2) * m.get(2, 1),
      this.get(0, 0) * m.get(0, 2) + this.get(0, 1) * m.get(1, 2) + this.get(0, 2) * m.get(2, 2),

      this.get(1, 0) * m.get(0, 0) + this.get(1, 1) * m.get(1, 0) + this.get(1, 2) * m.get(2, 0),
      this.get(1, 0) * m.get(0, 1) + this.get(1, 1) * m.get(1, 1) + this.get(1, 2) * m.get(2, 1),
      this.get(1, 0) * m.get(0, 2) + this.get(1, 1) * m.get(1, 2) + this.get(1, 2) * m.get(2, 2),

      this.get(2, 0) * m.get(0, 0) + this.get(2, 1) * m.get(1, 0) + this.get(2, 2) * m.get(2, 0),
      this.get(2, 0) * m.get(0, 1) + this.get(2, 1) * m.get(1, 1) + this.get(2, 2) * m.get(2, 1),
      this.get(2, 0) * m.get(0, 2) + this.get(2, 1) * m.get(1, 2) + this.get(2, 2) * m.get(2, 2)
    )
  }

  /** Returns a new matrix that is the inverse of this matrix. */
  inverse(): Mat3f {
    const det =
      this.get(0, 0) * (this.get(1, 1) * this.get(2, 2) - this.get(1, 2) * this.get(2, 1)) -
      this.get(0, 1) * (this.get(1, 0) * this.get(2, 2) - this.get(1, 2) * this.get(2, 0)) +
      this.get(0, 2) * (this.get(1, 0) * this.get(2, 1) - this.get(1, 1) * this.get(2, 0))
    if (det === 0) {
      throw new Error('Matrix is not invertible')
    }
    return new Mat3f(
      (this.get(1, 1) * this.get(2, 2) - this.get(1, 2) * this.get(2, 1)) / det,
      (this.get(0, 2) * this.get(2, 1) - this.get(0, 1) * this.get(2, 2)) / det,
      (this.get(0, 1) * this.get(1, 2) - this.get(0, 2) * this.get(1, 1)) / det,
      (this.get(1, 2) * this.get(2, 0) - this.get(1, 0) * this.get(2, 2)) / det,
      (this.get(0, 0) * this.get(2, 2) - this.get(0, 2) * this.get(2, 0)) / det,
      (this.get(0, 2) * this.get(1, 0) - this.get(0, 0) * this.get(1, 2)) / det,
      (this.get(1, 0) * this.get(2, 2) - this.get(1, 2) * this.get(2, 0)) / det,
      (this.get(0, 1) * this.get(2, 0) - this.get(0, 0) * this.get(2, 1)) / det,
      (this.get(0, 0) * this.get(1, 1) - this.get(0, 1) * this.get(1, 0)) / det
    )
  }
}

export { Vec2f, Mat3f }
