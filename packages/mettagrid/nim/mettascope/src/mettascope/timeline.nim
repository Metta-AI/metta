import
  std/[times, math],
  boxy, vmath, windy, silky,
  common, panels, actions, objectinfo

const
  TraceWidth = 0.54 / 2
  ScrubberColor = parseHtmlColor("#1D1D1D").rgbx

var
  # Drag state.
  scrubberActive = false
  minimapActive = false
  lastFrameTime: float64 = epochTime()

proc onRequestPython*() =
  ## Called before requesting Python to process the next step.
  processActions()

proc playControls*() =
  let now = epochTime()
  let deltaTime = now - lastFrameTime

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
    stepFloat += playSpeed * deltaTime
    case playMode:
    of Historical:
      if stepFloat >= replay.maxSteps.float32:
        # Loop back to the start.
        stepFloat -= replay.maxSteps.float32
    of Realtime:
      if stepFloat >= replay.maxSteps.float32:
        # Requesting more steps from Python.
        requestPython = true
        stepFloat = replay.maxSteps.float32 - 1
    step = stepFloat.int
    step = step.clamp(0, replay.maxSteps - 1)

  if window.buttonPressed[KeyLeftBracket]:
    step -= 1
    step = clamp(step, 0, replay.maxSteps - 1)
    stepFloat = step.float32
  if window.buttonPressed[KeyRightBracket]:
    step += 1
    if playMode == Realtime and step >= replay.maxSteps:
      requestPython = true
      step = replay.maxSteps - 1
    step = clamp(step, 0, replay.maxSteps - 1)
    stepFloat = step.float32
  # Fire onStepChanged once and only once when step changes.
  if step != previousStep:
    previousStep = step

  lastFrameTime = now

proc drawTimeline*(pos, size: Vec2) =
  ribbon(pos, size, ScrubberColor):
    let prevStepFloat = stepFloat
    scrubber("timeline", stepFloat, 0, replay.maxSteps.float32 - 1)
    if prevStepFloat != stepFloat:
      step = stepFloat.round.int
