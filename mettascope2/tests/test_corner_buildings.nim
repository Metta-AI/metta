import ../src/tribal/tribal, vmath

# Test that the corner buildings are created properly
proc testCornerBuildings() =
  echo "Creating new environment with corner buildings..."
  
  let env = newEnvironment()
  
  echo "Environment created successfully!"
  echo "\nMap visualization:"
  echo env.render()
  
  # Check for corner buildings
  var foundArmory = false
  var foundForge = false
  var foundClayOven = false
  var foundWeavingLoom = false
  
  for thing in env.things:
    case thing.kind:
    of Armory:
      foundArmory = true
      echo "Found Armory at position: ", thing.pos.x, ", ", thing.pos.y
    of Forge:
      foundForge = true
      echo "Found Forge at position: ", thing.pos.x, ", ", thing.pos.y
    of ClayOven:
      foundClayOven = true
      echo "Found Clay Oven at position: ", thing.pos.x, ", ", thing.pos.y
    of WeavingLoom:
      foundWeavingLoom = true
      echo "Found Weaving Loom at position: ", thing.pos.x, ", ", thing.pos.y
    else:
      discard
  
  echo "\nCorner buildings found:"
  echo "- Armory: ", foundArmory
  echo "- Forge: ", foundForge
  echo "- Clay Oven: ", foundClayOven
  echo "- Weaving Loom: ", foundWeavingLoom
  
  if foundArmory and foundForge and foundClayOven and foundWeavingLoom:
    echo "\n✓ All corner buildings created successfully!"
  else:
    echo "\n✗ Some corner buildings were not created"

when isMainModule:
  testCornerBuildings()