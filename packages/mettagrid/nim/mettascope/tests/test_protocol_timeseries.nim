## Unit tests for protocol time-series parsing and lookup
##
## Tests verify getProtocolsAt() returns correct protocols at any step,
## handling clip/unclip/re-clip cycles with different unclip protocols.

import std/[unittest, json, sequtils]
import mettascope/replays

suite "Protocol Time-Series Parsing and Lookup":
  
  # Helper to create a Protocol object
  proc makeProtocol(inputs: seq[(int, int)], outputs: seq[(int, int)], cooldown: int = 0, minAgents: int = 0): Protocol =
    result.minAgents = minAgents
    result.cooldown = cooldown
    result.vibes = @[]
    result.inputs = inputs.mapIt(ItemAmount(itemId: it[0], count: it[1]))
    result.outputs = outputs.mapIt(ItemAmount(itemId: it[0], count: it[1]))
  
  # Normal protocol: energy(0) -> silicon(4)
  let normalProtocol = makeProtocol(@[(0, 20)], @[(4, 15)])
  # Unclip protocol A: requires decoder(6)
  let unclipProtocolA = makeProtocol(@[(6, 1)], @[], cooldown = 1)
  # Unclip protocol B: requires scrambler(9)
  let unclipProtocolB = makeProtocol(@[(9, 1)], @[], cooldown = 1)
  # Unclip protocol C: requires modulator(7)
  let unclipProtocolC = makeProtocol(@[(7, 1)], @[], cooldown = 1)
  
  test "getProtocolsAt with empty time series returns protocols field":
    ## When protocolTimeSeries is empty, fall back to protocols field
    var entity = Entity()
    entity.protocols = @[normalProtocol]
    entity.protocolTimeSeries = @[]
    
    check entity.getProtocolsAt(0) == @[normalProtocol]
    check entity.getProtocolsAt(100) == @[normalProtocol]
    check entity.getProtocolsAt(999) == @[normalProtocol]
  
  test "2-entry case: clipped once, stays clipped":
    ## Timeline: step 0 = normal, step 100 = clipped
    var entity = Entity()
    entity.protocolTimeSeries = @[
      (step: 0, protocols: @[normalProtocol]),
      (step: 100, protocols: @[unclipProtocolA])
    ]
    
    # Before clip
    check entity.getProtocolsAt(0) == @[normalProtocol]
    check entity.getProtocolsAt(50) == @[normalProtocol]
    check entity.getProtocolsAt(99) == @[normalProtocol]
    
    # At and after clip
    check entity.getProtocolsAt(100) == @[unclipProtocolA]
    check entity.getProtocolsAt(150) == @[unclipProtocolA]
    check entity.getProtocolsAt(999) == @[unclipProtocolA]
  
  test "3-entry case: clipped then unclipped (final state unclipped)":
    ## Timeline: step 0 = normal, step 100 = clipped, step 200 = unclipped
    var entity = Entity()
    entity.protocolTimeSeries = @[
      (step: 0, protocols: @[normalProtocol]),
      (step: 100, protocols: @[unclipProtocolA]),
      (step: 200, protocols: @[normalProtocol])
    ]
    
    # Before first clip
    check entity.getProtocolsAt(0) == @[normalProtocol]
    check entity.getProtocolsAt(99) == @[normalProtocol]
    
    # While clipped
    check entity.getProtocolsAt(100) == @[unclipProtocolA]
    check entity.getProtocolsAt(150) == @[unclipProtocolA]
    check entity.getProtocolsAt(199) == @[unclipProtocolA]
    
    # After unclip - back to normal
    check entity.getProtocolsAt(200) == @[normalProtocol]
    check entity.getProtocolsAt(300) == @[normalProtocol]
    check entity.getProtocolsAt(999) == @[normalProtocol]
  
  test "4-entry case: clip, unclip, re-clip with different unclip protocol":
    ## Timeline: step 0 = normal, step 100 = clipped(A), step 200 = unclipped, step 300 = clipped(B)
    var entity = Entity()
    entity.protocolTimeSeries = @[
      (step: 0, protocols: @[normalProtocol]),
      (step: 100, protocols: @[unclipProtocolA]),  # First clip: decoder
      (step: 200, protocols: @[normalProtocol]),
      (step: 300, protocols: @[unclipProtocolB])   # Second clip: scrambler
    ]
    
    # Before first clip
    check entity.getProtocolsAt(0) == @[normalProtocol]
    
    # First clip period - requires decoder(6)
    check entity.getProtocolsAt(100) == @[unclipProtocolA]
    check entity.getProtocolsAt(100)[0].inputs[0].itemId == 6  # decoder
    
    # Unclipped period
    check entity.getProtocolsAt(200) == @[normalProtocol]
    check entity.getProtocolsAt(250) == @[normalProtocol]
    
    # Second clip period - requires scrambler(9), NOT decoder
    check entity.getProtocolsAt(300) == @[unclipProtocolB]
    check entity.getProtocolsAt(300)[0].inputs[0].itemId == 9  # scrambler
    check entity.getProtocolsAt(500) == @[unclipProtocolB]
  
  test "5-entry case: multiple clip/unclip cycles with all different protocols":
    ## 3 different unclip protocols across cycles
    var entity = Entity()
    entity.protocolTimeSeries = @[
      (step: 0, protocols: @[normalProtocol]),
      (step: 50, protocols: @[unclipProtocolA]),
      (step: 100, protocols: @[normalProtocol]),
      (step: 150, protocols: @[unclipProtocolB]),
      (step: 200, protocols: @[normalProtocol]),
      (step: 250, protocols: @[unclipProtocolC])
    ]
    
    # Check each transition point
    check entity.getProtocolsAt(0)[0].outputs.len > 0      # normal has outputs
    check entity.getProtocolsAt(50)[0].inputs[0].itemId == 6   # decoder
    check entity.getProtocolsAt(100)[0].outputs.len > 0    # normal
    check entity.getProtocolsAt(150)[0].inputs[0].itemId == 9  # scrambler
    check entity.getProtocolsAt(200)[0].outputs.len > 0    # normal
    check entity.getProtocolsAt(250)[0].inputs[0].itemId == 7  # modulator
    check entity.getProtocolsAt(999)[0].inputs[0].itemId == 7  # still modulator
  
  test "getProtocolsAt before first entry returns empty":
    ## Edge case: first entry at step > 0
    var entity = Entity()
    entity.protocolTimeSeries = @[
      (step: 10, protocols: @[normalProtocol])  # First entry at step 10, not 0
    ]
    
    # Step before first entry - return empty
    check entity.getProtocolsAt(0).len == 0
    check entity.getProtocolsAt(5).len == 0
    check entity.getProtocolsAt(9).len == 0
    
    # At and after first entry
    check entity.getProtocolsAt(10) == @[normalProtocol]
    check entity.getProtocolsAt(100) == @[normalProtocol]

suite "Protocol Time-Series JSON Parsing":
  
  test "Parse time-series format stores all entries":
    ## Verify time-series format populates protocolTimeSeries with all entries
    let jsonStr = """
    {
      "version": 2,
      "num_agents": 0,
      "max_steps": 500,
      "map_size": [10, 10],
      "type_names": ["silicon_extractor"],
      "action_names": ["noop"],
      "item_names": ["energy", "decoder", "scrambler"],
      "group_names": ["default"],
      "objects": [
        {
          "id": 1,
          "type_name": "silicon_extractor",
          "location": [[0, [0, 0]]],
          "orientation": [[0, 0]],
          "inventory": [[0, []]],
          "inventory_max": 10,
          "color": [[0, 0]],
          "is_clipped": [[0, false], [100, true], [200, false], [300, true]],
          "is_clip_immune": [[0, false]],
          "uses_count": [[0, 0]],
          "max_uses": 0,
          "allow_partial_usage": false,
          "protocols": [
            [0, [{"minAgents": 0, "vibes": [], "inputs": [[0, 20]], "outputs": [[4, 15]], "cooldown": 0}]],
            [100, [{"minAgents": 0, "vibes": [], "inputs": [[1, 1]], "outputs": [], "cooldown": 1}]],
            [200, [{"minAgents": 0, "vibes": [], "inputs": [[0, 20]], "outputs": [[4, 15]], "cooldown": 0}]],
            [300, [{"minAgents": 0, "vibes": [], "inputs": [[2, 1]], "outputs": [], "cooldown": 1}]]
          ]
        }
      ]
    }
    """
    
    let replay = loadReplayString(jsonStr, "test.json")
    check replay != nil
    check replay.objects.len == 1
    
    let entity = replay.objects[0]
    
    # Check that ALL 4 entries are stored in protocolTimeSeries
    check entity.protocolTimeSeries.len == 4
    check entity.protocolTimeSeries[0].step == 0
    check entity.protocolTimeSeries[1].step == 100
    check entity.protocolTimeSeries[2].step == 200
    check entity.protocolTimeSeries[3].step == 300
    
    # Verify correct protocols at each step
    check entity.getProtocolsAt(0)[0].outputs.len == 1        # normal
    check entity.getProtocolsAt(100)[0].inputs[0].itemId == 1 # decoder (first clip)
    check entity.getProtocolsAt(200)[0].outputs.len == 1      # normal (unclipped)
    check entity.getProtocolsAt(300)[0].inputs[0].itemId == 2 # scrambler (second clip)
  
  test "Parse direct array format sets empty protocolTimeSeries":
    ## Direct array format should use protocols field, not protocolTimeSeries
    let jsonStr = """
    {
      "version": 2,
      "num_agents": 0,
      "max_steps": 500,
      "map_size": [10, 10],
      "type_names": ["silicon_extractor"],
      "action_names": ["noop"],
      "item_names": ["energy"],
      "group_names": ["default"],
      "objects": [
        {
          "id": 1,
          "type_name": "silicon_extractor",
          "location": [[0, [0, 0]]],
          "orientation": [[0, 0]],
          "inventory": [[0, []]],
          "inventory_max": 10,
          "color": [[0, 0]],
          "is_clipped": [[0, false]],
          "is_clip_immune": [[0, false]],
          "uses_count": [[0, 0]],
          "max_uses": 0,
          "allow_partial_usage": false,
          "protocols": [
            {"minAgents": 0, "vibes": [], "inputs": [[0, 20]], "outputs": [[4, 15]], "cooldown": 0}
          ]
        }
      ]
    }
    """
    
    let replay = loadReplayString(jsonStr, "test.json")
    check replay != nil
    
    let entity = replay.objects[0]
    
    # Direct array format: protocolTimeSeries empty, protocols populated
    check entity.protocolTimeSeries.len == 0
    check entity.protocols.len == 1
    check entity.protocols[0].inputs[0].itemId == 0
    
    # getProtocolsAt falls back to protocols field
    check entity.getProtocolsAt(0) == entity.protocols
    check entity.getProtocolsAt(999) == entity.protocols
