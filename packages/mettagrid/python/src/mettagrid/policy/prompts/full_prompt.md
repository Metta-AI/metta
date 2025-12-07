{{BASIC_INFO}}

{{OBSERVABLE}}

=== DECISION PRIORITY ===

1. Check ADJACENT TILES to see what's around you
2. If adjacent to a useful object:
   - Set the right vibe if needed (heart_a for assembler, heart_b for chest)
   - Move INTO the object to use it
3. If not adjacent to anything useful:
   - Move toward the nearest useful object based on NEARBY AGENTS/OBJECTS info
   - Need resources? Find extractors
   - Have resources? Find assembler to craft hearts
   - Have heart? Find chest to deposit
   - Low energy? Find charger

⚠️ CRITICAL: OUTPUT FORMAT ⚠️
You MUST respond with ONLY a JSON object. NO other text, NO explanation, NO preamble.
If you write anything other than valid JSON, the game will crash.

REQUIRED FORMAT:
{"reasoning": "<brief thinking>", "action": "<action_name>"}

VALID ACTIONS: noop, move_north, move_south, move_east, move_west, change_vibe_heart_a, change_vibe_heart_b, change_vibe_default

Example response (copy this format EXACTLY):
{"reasoning": "Carbon extractor at x=1. Moving east.", "action": "move_east"}
