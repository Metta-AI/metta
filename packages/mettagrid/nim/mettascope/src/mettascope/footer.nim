import
  std/strutils,
  silky, chroma, vmath,
  common

const
  FooterColor = parseHtmlColor("#273646").rgbx

proc drawFooter*(pos, size: Vec2) =
  ribbon(pos, size, FooterColor):

    group(vec2(16, 16)):
      iconButton("ui/rewindToStart"):
        step = 0
      iconButton("ui/stepBack"):
        step -= 1
        step = clamp(step, 0, replay.maxSteps - 1)
      if play:
        iconButton("ui/pause"):
          play = false
      else:
        iconButton("ui/play"):
          play = true
      iconButton("ui/stepForward"):
        step += 1
        if step > replay.maxSteps - 1:
          echo "Requesting Python"
          requestPython = true
        step = clamp(step, 0, replay.maxSteps - 1)
      iconButton("ui/rewindToEnd"):
        step = replay.maxSteps - 1

    group(vec2(sk.size.x - 240, 16)):
      iconButton("ui/heart"):
        echo "heart"
      iconButton("ui/cloud"):
        echo "cloud"
      iconButton("ui/grid"):
        echo "grid"
      iconButton("ui/eye"):
        echo "eye"
      iconButton("ui/tack"):
        echo "tack"

# find "/UI/Main/GlobalFooter":
#   find "**/RewindToStart":
#     onClick:
#       step = 0
#   find "**/StepBack":
#     onClick:
#       step -= 1
#       step = clamp(step, 0, replay.maxSteps - 1)
#   find "**/Play":
#     onDisplay:
#       # TODO: Switch to pause icon when paused.
#       thisNode.setVariant("On", play)
#     onClick:
#       play = not play
#       thisNode.setVariant("On", play)
#   find "**/StepForward":
#     onClick:
#       step += 1
#       if step > replay.maxSteps - 1:
#         echo "Requesting Python"
#         requestPython = true
#       step = clamp(step, 0, replay.maxSteps - 1)
#   find "**/RewindToEnd":
#     onClick:
#       step = replay.maxSteps - 1

#   const speeds = [0.01, 0.05, 0.1, 0.5, 1, 5]
#   find "**/Speed?":
#     onDisplay:
#       var speedNum = parseInt($thisNode.name[^1]) - 1
#       thisNode.opacity =
#         if playSpeed < speeds[speedNum] - 0.00001:
#           0.5
#         else:
#           1
#     onClick:
#       var speedNum = parseInt($thisNode.name[^1]) - 1
#       playSpeed = speeds[speedNum]

#   find "**/Tack":
#     onDisplay:
#       thisNode.setVariant("On", settings.lockFocus)
#     onClick:
#       settings.lockFocus = not settings.lockFocus
#   find "**/Heart":
#     onDisplay:
#       thisNode.setVariant("On", settings.showResources)
#     onClick:
#       settings.showResources = not settings.showResources
#   find "**/Grid":
#     onDisplay:
#       thisNode.setVariant("On", settings.showGrid)
#     onClick:
#       settings.showGrid = not settings.showGrid
#   find "**/Eye":
#     onDisplay:
#       thisNode.setVariant("On", settings.showVisualRange)
#     onClick:
#       settings.showVisualRange = not settings.showVisualRange
#   find "**/Cloud":
#     onDisplay:
#       thisNode.setVariant("On", settings.showFogOfWar)
#     onClick:
#       settings.showFogOfWar = not settings.showFogOfWar
