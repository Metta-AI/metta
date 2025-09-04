import
  std/[strformat],
  boxy, vmath, windy, fidget2/[hybridrender, common], fidget2,
  common, panels, sim, actions, utils, ui

find "/UI/Main/GlobalFooter":
  find "**/RewindToStart":
    onClick:
      step = 0
  find "**/StepBack":
    onClick:
      step -= 1
      step = clamp(step, 0, replay.maxSteps - 1)
  find "**/Play":
    onClick:
      echo "Clicked: Play"
      play = not play
  find "**/StepForward":
    onClick:
      step += 1
      step = clamp(step, 0, replay.maxSteps - 1)
  find "**/RewindToEnd":
    onClick:
      step = replay.maxSteps - 1

  find "**/Speed1":
    onClick:
      playSpeed = 0.01
  find "**/Speed2":
    onClick:
      playSpeed = 0.05
  find "**/Speed3":
    onClick:
      playSpeed = 0.1
  find "**/Speed4":
    onClick:
      playSpeed = 0.5
  find "**/Speed5":
    onClick:
      playSpeed = 1
  find "**/Speed6":
    onClick:
      playSpeed = 5

  find "**/Tack":
    onClick:
      settings.lockFocus = not settings.lockFocus
  find "**/Heart":
    onClick:
      settings.showResources = not settings.showResources
  find "**/Grid":
    onClick:
      settings.showGrid = not settings.showGrid
  find "**/Eye":
    onClick:
      settings.showVisualRange = not settings.showVisualRange
  find "**/Cloud":
    onClick:
      settings.showFogOfWar = not settings.showFogOfWar
