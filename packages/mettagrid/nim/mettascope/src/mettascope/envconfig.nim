import
  std/[json],
  fidget2,
  common

proc drawEnvConfig*(panel: Panel) =
  ## Draw the environment configuration JSON as text inside the panel.
  var text = "No configuration loaded."
  if not replay.isNil and not replay.mgConfig.isNil:
    text = pretty(replay.mgConfig)

  let wrapper = find("**/ConfigTextWrapper")
  wrapper.clipsContent = true
  wrapper.overflowDirection = VerticalScrolling

  let cfgText = find("**/ConfigText")

  if cfgText.text != text:
    cfgText.text = text
