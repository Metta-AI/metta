## Headless driver for profiling the Nim environment without the renderer.
## Build with profiling enabled, e.g.:
##   nim r --nimcache:./nimcache --profiler:on --stackTrace:on profile_env.nim

import nimprof
import std/random
import src/environment

when isMainModule:
  var env = newEnvironment()
  var actions: array[MapAgents, uint8]
  var rng = initRand(42)
  for step in 0 ..< 500:
    for i in 0 ..< MapAgents:
      let verb = uint8(rng.rand(0 .. 6))
      let argument = uint8(rng.rand(0 .. 7))
      actions[i] = encodeAction(verb, argument)
    env.step(addr actions)
