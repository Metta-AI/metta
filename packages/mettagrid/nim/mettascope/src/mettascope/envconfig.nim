import
  std/[json],
  boxy, chroma, fidget2/[hybridrender, common],
  common, panels, utils

proc drawEnvConfig*(panel: Panel) =
  ## Draw the environment configuration JSON as text inside the panel.
  bxy.drawRect(
    rect = rect(0, 0, panel.rect.w.float32, panel.rect.h.float32),
    color = color(0, 0, 0, 0.6)
  )

  var text = "No configuration loaded."
  if not replay.isNil and not replay.mgConfig.isNil:
    # Pretty-print the mg_config JSON.
    text = pretty(replay.mgConfig)

  # Render the JSON text with some padding (node-local coordinates).
  let transform = translate(vec2(8, 8))
  bxy.drawText(
    "env-config-text",
    transform,
    utils.typeface,
    text,
    12.0'f,
    color(1, 1, 1, 1.0)
  )
