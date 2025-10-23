import math, random

type
  Perlin2D* = object
    perm: array[512, int]  # permutation table duplicated

proc xorshift32(seed: var uint32): uint32 =
  ## Simple PRNG for deterministic shuffling.
  var x = seed
  x = x xor (x shl 13)
  x = x xor (x shr 17)
  x = x xor (x shl 5)
  seed = x
  result = x

proc initPerlin2D*(seed: uint32 = 0'u32): Perlin2D =
  ## Build a permutation table using the given seed.
  var base: array[256, int]
  for i in 0 .. 255:
    base[i] = i
  var s =
    if seed == 0:
      0x9E3779B9'u32
    else:
      seed
  # Fisher–Yates shuffle with xorshift32.
  for i in countdown(255, 1):
    let r = int(xorshift32(s) mod uint32(i+1))
    swap base[i], base[r]
  for i in 0..255:
    result.perm[i] = base[i]
    result.perm[i+256] = base[i]

proc fade(t: float32): float32 =
  ## Fade function.
  t*t*t*(t*(t*6'f32 - 15'f32) + 10'f32)

proc lerp(a, b, t: float32): float32 =
  ## Linear interpolation.
  a + t*(b - a)

proc grad(hash: int, x, y: float32): float32 =
  ## 8 gradient directions
  case (hash and 7)
  of 0:  x + y
  of 1: -x + y
  of 2:  x - y
  of 3: -x - y
  of 4:  x
  of 5: -x
  of 6:  y
  else: -y

proc noise*(p: Perlin2D, x, y: float32): float32 =
  ## Classic 2D Perlin noise in [-1, 1].
  let xi = (floor(x).int) and 255
  let yi = (floor(y).int) and 255
  let xf = x - floor(x).float32
  let yf = y - floor(y).float32

  let u = fade(xf)
  let v = fade(yf)

  let aa = p.perm[(p.perm[xi]     + yi    ) and 255]
  let ab = p.perm[(p.perm[xi]     + yi + 1) and 255]
  let ba = p.perm[(p.perm[xi + 1] + yi    ) and 255]
  let bb = p.perm[(p.perm[xi + 1] + yi + 1) and 255]

  let x1 = lerp(grad(aa, xf    , yf    ), grad(ba, xf - 1, yf    ), u)
  let x2 = lerp(grad(ab, xf    , yf - 1), grad(bb, xf - 1, yf - 1), u)
  result = lerp(x1, x2, v)
