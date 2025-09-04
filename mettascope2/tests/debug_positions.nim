import ../src/mettascope/tribal
import strformat

# Create environment and check positions
var env = newEnvironment()

echo "Temple and Clippy positions:"
for thing in env.things:
  if thing.kind == Temple:
    echo fmt"Temple at ({thing.pos.x}, {thing.pos.y})"
  elif thing.kind == Clippy:
    echo fmt"Clippy at ({thing.pos.x}, {thing.pos.y})"

# Check if they're at the same position (which would hide Clippys)
for temple in env.things:
  if temple.kind == Temple:
    for clippy in env.things:
      if clippy.kind == Clippy:
        if temple.pos.x == clippy.pos.x and temple.pos.y == clippy.pos.y:
          echo fmt"WARNING: Clippy at same position as Temple! ({temple.pos.x}, {temple.pos.y})"