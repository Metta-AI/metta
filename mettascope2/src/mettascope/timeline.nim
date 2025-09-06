import
  std/[strformat],
  boxy, vmath, windy, fidget2/[hybridrender, common],
  common, panels, sim, actions, utils

const
  BgColor = parseHtmlColor("#1D1D1D")

proc playControls*() =
  if window.buttonPressed[KeySpace]:
    play = not play
    stepFloat = step.float32
  if window.buttonPressed[KeyMinus]:
    playSpeed *= 0.5
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true
  if window.buttonPressed[KeyEqual]:
    playSpeed *= 2
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true

  if play:
    stepFloat += playSpeed
    if stepFloat >= replay.maxSteps.float32:
      stepFloat -= replay.maxSteps.float32
    step = stepFloat.int
    step = step.clamp(0, replay.maxSteps - 1)

  if window.buttonPressed[KeyLeftBracket]:
    step -= 1
    step = clamp(step, 0, replay.maxSteps - 1)
    stepFloat = step.float32
    echo "step: ", step
  if window.buttonPressed[KeyRightBracket]:
    step += 1
    step = clamp(step, 0, replay.maxSteps - 1)
    stepFloat = step.float32
    echo "step: ", step
