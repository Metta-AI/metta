import
  std/[json, sequtils, strformat, strutils],
  mettascope/validation

proc getMinimalReplay(fileName: string = "sample.json.z"): JsonNode =
  ## Create a minimal valid replay dict per the spec.
  result = %*{
    "version": 3,
    "num_agents": 2,
    "max_steps": 100,
    "map_size": [10, 10],
    "file_name": "test replay file format",
    "type_names": ["agent", "assembler", "resource"],
    "action_names": ["move", "collect"],
    "item_names": ["wood", "stone"],
    "group_names": ["group1", "group2"],
    "reward_sharing_matrix": [[1, 0], [0, 1]],
    "objects": [
      {
        "id": 1,
        "type_name": "agent",
        "agent_id": 0,
        "is_agent": true,
        "vision_size": 11,
        "group_id": 0,
        # Time series fields (some single values, some arrays for testing)
        "location": [[0, [5, 5]], [1, [6, 5]], [2, [7, 5]]],
        "action_id": 0,
        "action_param": 0,
        "action_success": true,
        "current_reward": 0.0,
        "total_reward": 0.0,
        "freeze_remaining": 0,
        "is_frozen": false,
        "freeze_duration": 0,
        "orientation": 0,
        "inventory": [],
        "inventory_max": 10,
        "color": 0,
      },
      {
        "id": 2,
        "type_name": "agent",
        "agent_id": 1,
        "is_agent": true,
        "vision_size": 11,
        "group_id": 0,
        # Time series fields (mix of single values and arrays for testing)
        "location": [[0, [3, 3]], [5, [4, 3]]],
        "action_id": [[0, 1], [10, 0]],
        "action_param": 0,
        "action_success": [[0, false], [10, true]],
        "current_reward": 1.5,
        "total_reward": [[0, 0.0], [10, 1.5]],
        "freeze_remaining": 0,
        "is_frozen": false,
        "freeze_duration": 0,
        "orientation": 1,
        "inventory": [[0, []], [100, [[1, 1]]], [200, [[1, 2]]]],
        "inventory_max": 10,
        "color": 1,
      },
      {
        "id": 3,
        "type_name": "assembler",
        # Assembler-specific fields
        "protocols": [
          {
            "minAgents": 0,
            "vibes": [1, 2],  # Example vibes
            "inputs": [[0, 2]],  # 2 wood
            "outputs": [[1, 1]], # 1 stone
            "cooldown": 10
          }
        ],
        "cooldown_remaining": 0,
        "cooldown_duration": 10,
        "is_clipped": false,
        "is_clip_immune": false,
        "uses_count": 0,
        "max_uses": 100,
        "allow_partial_usage": true,
        # Common fields
        "location": [[0, [1, 1]]],
        "orientation": 0,
        "inventory": [],
        "inventory_max": 50,
        "color": 2,
      },
    ]
  }



block schema_validation:
  block valid_replay:
    let replay = getMinimalReplay()
    let issues = validateReplay(replay)
    doAssert issues.len == 0, &"Valid replay should have no issues, but got: {issues}"
    echo "✓ Valid replay schema passes validation"

  block invalid_version:
    var replay = getMinimalReplay()
    replay["version"] = %*1
    let issues = validateReplay(replay)
    doAssert issues.len > 0, "Should have validation issues"
    doAssert issues[0].message.contains("'version' must equal 3"), &"Unexpected issue: {issues[0].message}"
    echo "✓ Invalid version properly rejected"

  block invalid_num_agents:
    var replay = getMinimalReplay()
    replay["num_agents"] = %*(-1)
    let issues = validateReplay(replay)
    doAssert issues.len > 0, "Should have validation issues"
    doAssert issues.anyIt(it.message.contains("'num_agents' must be non-negative")), &"Expected non-negative validation error, got: {issues}"
    echo "✓ Invalid num_agents properly rejected"

  block invalid_map_size:
    var replay = getMinimalReplay()
    replay["map_size"] = %*[0, 5]
    let issues = validateReplay(replay)
    doAssert issues.len > 0, "Should have validation issues"
    doAssert issues.anyIt(it.message.contains("'map_size[0]' must be positive")), &"Expected positive validation error, got: {issues}"
    echo "✓ Invalid map_size properly rejected"

  block empty_objects_valid_when_no_agents:
    var replay = getMinimalReplay()
    replay["num_agents"] = %*0
    replay["objects"] = %*[]
    replay.delete("reward_sharing_matrix")
    let issues = validateReplay(replay)
    doAssert issues.len == 0, &"Replay with map but empty objects should be valid when num_agents=0, but got: {issues}"
    echo "✓ Empty objects array valid when no agents expected"

  block missing_required_fields_gracefully_handled:
    var replay = getMinimalReplay()
    # Remove type_name from first object (required field)
    replay["objects"][0].delete("type_name")
    let issues = validateReplay(replay)
    # Should have issues but not crash
    doAssert issues.len > 0, "Should have validation issues for missing required fields"
    doAssert issues.anyIt(it.message.contains("must have either") or it.message.contains("is missing (required)")), &"Expected missing fields validation error, got: {issues}"
    echo "✓ Missing required fields handled gracefully without crashing"

  block invalid_location_format_gracefully_handled:
    var replay = getMinimalReplay()
    # Set invalid location format (stepData is not an array of length 2)
    replay["objects"][0]["location"] = %*["invalid_step_data", [1, [2, 3]]]
    let issues = validateReplay(replay)
    # Should have issues but not crash
    doAssert issues.len > 0, "Should have validation issues for invalid location format"
    doAssert issues.anyIt(it.message.contains("items must be [step, [x, y]] pairs")), &"Expected location format validation error, got: {issues}"
    echo "✓ Invalid location format handled gracefully without crashing"
