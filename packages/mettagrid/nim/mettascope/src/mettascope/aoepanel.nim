## AOE panel displays checkboxes to toggle AOE overlays for each team.

import
  std/[json, strformat, sets],
  bumpy, silky, chroma, vmath, windy,
  common, panels, replays

const
  UnalignedId* = -1

proc getNumCommons*(): int =
  ## Get the number of commons from the replay config.
  if replay.isNil or replay.mgConfig.isNil:
    return 0
  if "game" notin replay.mgConfig or "commons" notin replay.mgConfig["game"]:
    return 0
  let commonsArr = replay.mgConfig["game"]["commons"]
  if commonsArr.kind == JArray:
    return commonsArr.len
  return 0

proc getCommonsName*(commonsId: int): string =
  ## Get the commons name by ID from the mg_config.
  if replay.isNil or replay.mgConfig.isNil:
    return ""
  if "game" notin replay.mgConfig or "commons" notin replay.mgConfig["game"]:
    return ""
  let commonsArr = replay.mgConfig["game"]["commons"]
  if commonsArr.kind != JArray or commonsId < 0 or commonsId >= commonsArr.len:
    return ""
  let commonsConfig = commonsArr[commonsId]
  if commonsConfig.kind == JObject and "name" in commonsConfig:
    return commonsConfig["name"].getStr
  return ""

proc getAoeColor*(commonsId: int): Color =
  ## Get color for each commons ID: Cogs (0) = green, Clips (1) = red, Neutral = grey.
  case commonsId
  of UnalignedId: color(0.5, 0.5, 0.5, 0.4)  # Neutral = grey
  of 0: color(0.2, 0.8, 0.2, 0.4)            # Cogs = green
  of 1: color(0.9, 0.2, 0.2, 0.4)            # Clips = red
  else: color(0.5, 0.5, 0.5, 0.4)            # Others = grey

proc drawAoeToggle(label: string, enabled: bool, tintColor: Color): bool =
  ## Draw a toggle button with a color indicator. Returns true if clicked.
  let boxSize = 20.0f
  let spacing = 8.0f
  let textSize = sk.getTextSize("Default", label)
  let totalWidth = boxSize + spacing + textSize.x
  let totalHeight = max(boxSize, textSize.y)
  let startPos = sk.at
  # Draw color box.
  let tintRgbx = rgbx(
    (tintColor.r * 255).uint8,
    (tintColor.g * 255).uint8,
    (tintColor.b * 255).uint8,
    255
  )
  sk.drawRect(startPos, vec2(boxSize, boxSize), tintRgbx)
  # Draw check mark if enabled.
  if enabled:
    sk.drawRect(startPos + vec2(4, 4), vec2(boxSize - 8, boxSize - 8), rgbx(255, 255, 255, 255))
  # Draw label.
  discard sk.drawText("Default", label, startPos + vec2(boxSize + spacing, 2), rgbx(255, 255, 255, 255))
  # Advance the layout.
  sk.advance(vec2(0, totalHeight + 4))
  # Handle click.
  let clickRect = bumpy.rect(startPos, vec2(totalWidth, totalHeight))
  if mouseInsideClip(clickRect) and window.buttonPressed[MouseLeft]:
    return true
  return false

proc drawAoePanel*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  ## Draw the AOE panel with checkboxes for each team.
  frame(frameId, contentPos, contentSize):
    sk.at += vec2(8, 8)
    h1text("AOE Overlays")
    sk.advance(vec2(0, 8))
    let numCommons = getNumCommons()
    # Draw toggle for unaligned objects.
    let unalignedEnabled = UnalignedId in settings.aoeEnabledCommons
    let unalignedColor = getAoeColor(UnalignedId)
    if drawAoeToggle("Unaligned", unalignedEnabled, unalignedColor):
      if UnalignedId in settings.aoeEnabledCommons:
        settings.aoeEnabledCommons.excl(UnalignedId)
      else:
        settings.aoeEnabledCommons.incl(UnalignedId)
      viewStateChanged = true
    # Draw toggle for each team.
    for i in 0 ..< numCommons:
      let commonsName = getCommonsName(i)
      let displayName = if commonsName.len > 0: commonsName else: &"Team {i}"
      let enabled = i in settings.aoeEnabledCommons
      let teamColor = getAoeColor(i)
      if drawAoeToggle(displayName, enabled, teamColor):
        if i in settings.aoeEnabledCommons:
          settings.aoeEnabledCommons.excl(i)
        else:
          settings.aoeEnabledCommons.incl(i)
        viewStateChanged = true
    # Add toggle all/none buttons.
    sk.advance(vec2(0, 12))
    group(vec2(0, 0), LeftToRight):
      button("All"):
        settings.aoeEnabledCommons.incl(UnalignedId)
        for i in 0 ..< numCommons:
          settings.aoeEnabledCommons.incl(i)
        viewStateChanged = true
      button("None"):
        settings.aoeEnabledCommons.clear()
        viewStateChanged = true

