import
  std/[strformat],
  boxy, vmath, windy, fidget2/[hybridrender, common],
  common, panels, sim, actions, utils, ui

const
  BgColor = parseHtmlColor("#273646")

proc drawHeader*(panel: Panel) =
  bxy.drawRect(
    rect = Rect(
      x: 0,
      y: 0,
      w: panel.rect.w.float32,
      h: panel.rect.h.float32
    ),
    color = BgColor
  )
  bxy.drawImage(
    "ui/logo",
    pos = vec2(0, 0),
  )
  bxy.drawImage(
    "ui/header-bg",
    pos = vec2(0, 0),
  )

  # Draw the title.
  bxy.drawText(
    "Mettascope Arena Basic",
    translate(vec2(64+16, 16)),
    typeface,
    "Mettascope Arena Basic",
    24,
    color(1, 1, 1, 1)
  )

  if drawIconButton(
    "ui/share",
    pos = vec2(panel.rect.w.float32 - (16 + 32)*1, 16)
  ):
    echo "Share"

  if drawIconButton(
    "ui/help",
    pos = vec2(panel.rect.w.float32 - (16 + 32)*2, 16)
  ):
    echo "Help"
