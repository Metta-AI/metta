## Integration tests for protocol time-series parsing
##
## Tests loadReplay with both direct array and time-series protocol formats.
## Test data: tests/data/replays/failing_replay.json.z (time-series)
## Test data: tests/data/replays/dinky7.json.z (direct array)

import std/[unittest, os]
import mettascope/replays

const testDataDir = currentSourcePath().parentDir() / "data" / "replays"

suite "Protocol Parsing Integration":
  
  test "Load replay with time-series protocols succeeds":
    let replayPath = testDataDir / "failing_replay.json.z"
    doAssert fileExists(replayPath), "Test file not found: " & replayPath
    
    let replay = loadReplay(replayPath)
    check replay != nil
    check replay.objects.len > 0
  
  test "Time-series protocols stored in protocolTimeSeries":
    ## Object 785 (silicon_extractor): step 0 normal, step 510 clipped
    let replayPath = testDataDir / "failing_replay.json.z"
    let replay = loadReplay(replayPath)
    
    # Find object 785
    var obj785: Entity = nil
    for obj in replay.objects:
      if obj.id == 785:
        obj785 = obj
        break
    
    check obj785 != nil
    check obj785.typeName == "silicon_extractor"
    
    # Time-series format: should have protocolTimeSeries populated
    check obj785.protocolTimeSeries.len == 2
    check obj785.protocolTimeSeries[0].step == 0
    check obj785.protocolTimeSeries[1].step == 510
    
    # getProtocolsAt returns normal protocols before clip
    let normalProtos = obj785.getProtocolsAt(0)
    check normalProtos.len == 1
    check normalProtos[0].inputs.len == 1
    check normalProtos[0].inputs[0].itemId == 0   # energy
    check normalProtos[0].inputs[0].count == 20
    check normalProtos[0].outputs.len == 1
    check normalProtos[0].outputs[0].itemId == 4  # silicon
    check normalProtos[0].outputs[0].count == 15
    check normalProtos[0].cooldown == 0
    
    # getProtocolsAt returns unclip protocols after clip
    let unclipProtos = obj785.getProtocolsAt(510)
    check unclipProtos.len == 1
    check unclipProtos[0].inputs.len == 1
    check unclipProtos[0].inputs[0].itemId == 9  # scrambler
    check unclipProtos[0].inputs[0].count == 1
    check unclipProtos[0].outputs.len == 0  # No output when clipped
    check unclipProtos[0].cooldown == 1
    
    # Also works at steps between transitions
    check obj785.getProtocolsAt(100) == normalProtos  # Still normal
    check obj785.getProtocolsAt(509) == normalProtos  # Just before clip
    check obj785.getProtocolsAt(600) == unclipProtos  # After clip
  
  test "Germanium extractor getProtocolsAt returns correct protocols":
    ## Object 1222: step 0 has 4 protocols, step 161 clipped
    let replayPath = testDataDir / "failing_replay.json.z"
    let replay = loadReplay(replayPath)
    
    # Find object 1222
    var obj1222: Entity = nil
    for obj in replay.objects:
      if obj.id == 1222:
        obj1222 = obj
        break
    
    check obj1222 != nil
    check obj1222.typeName == "germanium_extractor"
    
    # Time-series format
    check obj1222.protocolTimeSeries.len == 2
    
    # Normal protocols at step 0 - 4 different min_agents variants
    let normalProtos = obj1222.getProtocolsAt(0)
    check normalProtos.len == 4
    check normalProtos[0].minAgents == 4
    check normalProtos[0].outputs[0].itemId == 3  # germanium
    check normalProtos[0].outputs[0].count == 5
    check normalProtos[3].minAgents == 0
    check normalProtos[3].outputs[0].count == 2
    
    # Unclip protocol at step 161 - needs modulator
    let unclipProtos = obj1222.getProtocolsAt(161)
    check unclipProtos.len == 1
    check unclipProtos[0].inputs[0].itemId == 7  # modulator
    check unclipProtos[0].outputs.len == 0
  
  test "Direct array protocols work (empty protocolTimeSeries)":
    let replayPath = testDataDir / "dinky7.json.z"
    doAssert fileExists(replayPath), "Test file not found: " & replayPath
    
    let replay = loadReplay(replayPath)
    check replay != nil
    
    # Find any object with protocols (direct array format)
    var foundWithProtocols = false
    for obj in replay.objects:
      if obj.protocols.len > 0:
        foundWithProtocols = true
        # Direct array format: protocolTimeSeries should be empty
        check obj.protocolTimeSeries.len == 0
        # getProtocolsAt should return protocols field
        check obj.getProtocolsAt(0) == obj.protocols
        check obj.getProtocolsAt(999) == obj.protocols
        break
    
    check foundWithProtocols
  
  test "Replay count of objects with time-series protocols":
    ## Regression test: failing_replay has 18 objects with time-series protocols
    let replayPath = testDataDir / "failing_replay.json.z"
    let replay = loadReplay(replayPath)
    
    var countWithTimeSeries = 0
    for obj in replay.objects:
      if obj.protocolTimeSeries.len > 0:
        countWithTimeSeries += 1
    
    # The failing_replay has 18 objects with time-series protocols
    check countWithTimeSeries == 18
