import
  std/[strformat],
  boxy, vmath, windy,
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

proc drawTimeline*(panel: Panel) =
  playControls()

  bxy.drawRect(
    rect = Rect(
      x: 0,
      y: 0,
      w: panel.rect.w.float32,
      h: panel.rect.h.float32
    ),
    color = BgColor
  )

  # Draw the scrubber bg.
  bxy.drawRect(
    rect = Rect(
      x: 16,
      y: 32,
      w: panel.rect.w.float32 - 32,
      h: 16
    ),
    color = parseHtmlColor("#717171")
  )

  var progress = step.float32 / replay.maxSteps.float32

  # Draw the progress bar.
  bxy.drawRect(
    rect = Rect(
      x: 16,
      y: 32,
      w: (panel.rect.w.float32 - 32) * progress,
      h: 16
    ),
    color = color(1, 1, 1, 1)
  )

  # Clicking on the timeline will set it.
  let box = Rect(
    x: 16,
    y: 32,
    w: panel.rect.w.float32 - 32,
    h: 16
  )
  if window.boxyMouse.vec2.overlaps(box):
    if window.buttonDown[MouseLeft]:
      let progress = (window.boxyMouse.vec2.x - 16) / (panel.rect.w.float32 - 32)
      step = int(progress * replay.maxSteps.float32)
      step = clamp(step, 0, replay.maxSteps - 1)
      stepFloat = step.float32
      echo "step: ", step
