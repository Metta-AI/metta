import
  std/strutils,
  silky, chroma, vmath,
  common

const
  FooterColor = parseHtmlColor("#273646").rgbx
  Speeds = [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]

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

    sk.at = pos + vec2(size.x - 240, 16)
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
