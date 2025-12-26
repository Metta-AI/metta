import
  std/[strutils, json],
  silky, chroma, vmath, windy, bumpy,
  common, replays

const
  FooterColor = parseHtmlColor("#273646").rgbx
  Speeds = [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
  AOE_ALL* = 999  ## Special value meaning "show all AOE"

proc getNumCommons(): int =
  ## Get the number of commons from the replay config.
  if replay.isNil or replay.mgConfig.isNil:
    return 0
  if "game" notin replay.mgConfig or "commons" notin replay.mgConfig["game"]:
    return 0
  let commonsArr = replay.mgConfig["game"]["commons"]
  if commonsArr.kind == JArray:
    return commonsArr.len
  return 0

proc cycleAOEState() =
  ## Cycle through AOE states: off → commons-0 → commons-1 → ... → all → off
  let numCommons = getNumCommons()
  if settings.showAOE == -1:
    # Off → first commons (or all if no commons)
    if numCommons > 0:
      settings.showAOE = 0
    else:
      settings.showAOE = AOE_ALL
  elif settings.showAOE >= 0 and settings.showAOE < numCommons - 1:
    # Next commons
    settings.showAOE += 1
  elif settings.showAOE == numCommons - 1 or settings.showAOE < AOE_ALL:
    # Last commons → all
    settings.showAOE = AOE_ALL
  else:
    # All → off
    settings.showAOE = -1

proc drawFooter*(pos, size: Vec2) =
  ribbon(pos, size, FooterColor):

    let pos = sk.pos
    let size = sk.size

    sk.at = pos + vec2(16, 16)
    group(vec2(0, 0), LeftToRight):
      clickableIcon("ui/rewindToStart", step == 0):
        step = 0
        stepFloat = step.float32
      clickableIcon("ui/stepBack", false):
        step -= 1
        step = clamp(step, 0, replay.maxSteps - 1)
        stepFloat = step.float32
      if play:
        clickableIcon("ui/pause", true):
          play = false
      else:
        clickableIcon("ui/play", false):
          play = true
      clickableIcon("ui/stepForward", false):
        step += 1
        if step > replay.maxSteps - 1:
          requestPython = true
        step = clamp(step, 0, replay.maxSteps - 1)
        stepFloat = step.float32
      clickableIcon("ui/rewindToEnd", step == replay.maxSteps - 1):
        step = replay.maxSteps - 1
        stepFloat = step.float32

    sk.at = pos + vec2(size.x/2 - 120, 16)
    group(vec2(0, 0), LeftToRight):
      for i, speed in Speeds:
        if i == 0:
          clickableIcon("ui/turtle", playSpeed >= speed):
            playSpeed = speed
        elif i == len(Speeds) - 1:
          clickableIcon("ui/rabbit", playSpeed >= speed):
            playSpeed = speed
        else:
          clickableIcon("ui/speed", playSpeed >= speed):
            playSpeed = speed

    sk.at = pos + vec2(size.x - 280, 16)
    var aoeIconPos: Vec2
    group(vec2(0, 0), LeftToRight):
      clickableIcon("ui/tack", settings.lockFocus):
        settings.lockFocus = not settings.lockFocus
      clickableIcon("ui/heart", settings.showResources):
        settings.showResources = not settings.showResources
      clickableIcon("ui/grid", settings.showGrid):
        settings.showGrid = not settings.showGrid
      clickableIcon("ui/eye", settings.showVisualRange):
        settings.showVisualRange = not settings.showVisualRange
      clickableIcon("ui/cloud", settings.showFogOfWar):
        settings.showFogOfWar = not settings.showFogOfWar
      # AOE toggle: left click cycles states, right click turns off
      aoeIconPos = sk.at
      clickableIcon("ui/target", settings.showAOE >= 0):
        cycleAOEState()
    # Check for right-click on the AOE icon
    if mouseInsideClip(bumpy.rect(aoeIconPos, vec2(32, 32))) and window.buttonReleased[MouseRight]:
      settings.showAOE = -1
