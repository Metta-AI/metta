## Test utilities to reduce boilerplate across test files
import ../src/mettascope/[tribal, clippy, terrain, placement, village]
import std/[strformat, tables]
import vmath

export tribal, clippy, terrain, placement, village, vmath, strformat

# Common test procedures
proc runSteps*(env: Environment, numSteps: int, actionType: uint8 = 0, actionArg: uint8 = 0) =
  ## Run simulation for specified steps with all agents doing same action
  for step in 1..numSteps:
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0 ..< MapAgents:
      actions[i] = [actionType, actionArg]
    env.step(actions.addr)

proc countEntities*(env: Environment): tuple[agents: int, clippys: int, temples: int, altars: int] =
  ## Count different entity types in environment
  result = (agents: 0, clippys: 0, temples: 0, altars: 0)
  for thing in env.things:
    case thing.kind:
    of Agent: result.agents += 1
    of Clippy: result.clippys += 1
    of Temple: result.temples += 1
    of Altar: result.altars += 1
    else: discard

proc countDeadAgents*(env: Environment): int =
  ## Count terminated agents
  for i in 0 ..< MapAgents:
    if env.terminated[i] == 1.0:
      result += 1

proc getTotalAltarHearts*(env: Environment): int =
  ## Get sum of all altar hearts
  for thing in env.things:
    if thing.kind == Altar:
      result += thing.hp

proc printEntityCounts*(env: Environment, label: string = "") =
  ## Print current entity counts
  let counts = env.countEntities()
  let dead = env.countDeadAgents()
  if label != "":
    echo label
  echo fmt"  Agents: {counts.agents} (dead: {dead})"
  echo fmt"  Clippys: {counts.clippys}"
  echo fmt"  Temples: {counts.temples}"
  echo fmt"  Altars: {counts.altars} (total hearts: {env.getTotalAltarHearts()})"

proc findNearbyPair*(env: Environment, kind1, kind2: ThingKind): bool =
  ## Check if any entities of two types are adjacent
  for thing1 in env.things:
    if thing1.kind == kind1:
      for thing2 in env.things:
        if thing2.kind == kind2:
          let dist = abs(thing1.pos.x - thing2.pos.x) + abs(thing1.pos.y - thing2.pos.y)
          if dist == 1:
            return true
  return false