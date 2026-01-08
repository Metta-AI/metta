## Minimal deterministic RNG used across the project.
## Avoids std/random so wasm builds do not depend on sysrand.

const DefaultSeed* = 0x9E3779B97F4A7C15'u64

type
  Rand* = object
    state: uint64

proc initRand*(seed: int): Rand =
  var s = uint64(seed)
  if s == 0:
    s = DefaultSeed
  result.state = s

proc mix(state: uint64): uint64 =
  var x = state
  x = x xor (x shl 13)
  x = x xor (x shr 7)
  x = x xor (x shl 17)
  x

proc next*(r: var Rand): uint64 =
  r.state = mix(r.state)
  r.state

proc randIntInclusive*(r: var Rand, a, b: int): int =
  if a >= b:
    return a
  let range = uint64(b - a + 1)
  a + int(next(r) mod range)

proc randIntExclusive*(r: var Rand, a, b: int): int =
  if a >= b:
    return a
  let range = uint64(b - a)
  a + int(next(r) mod range)

proc randFloat*(r: var Rand): float64 =
  const factor = 1.0 / float64(1'u64 shl 53)
  float64(next(r) shr 11) * factor

proc randChance*(r: var Rand, probability: float): bool =
  if probability <= 0: return false
  if probability >= 1: return true
  randFloat(r) < probability

proc sample*[T](r: var Rand, items: openArray[T]): T =
  if items.len == 0:
    raise newException(ValueError, "Cannot sample from empty sequence")
  items[randIntExclusive(r, 0, items.len)]
