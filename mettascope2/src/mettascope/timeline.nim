import
  std/[strformat],
  boxy, vmath, windy,
  common, panels, sim, actions, utils

const
  BgColor = parseHtmlColor("#1D1D1D")

proc drawTimeline*(panel: Panel) =
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

  var progress = 0.37

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
