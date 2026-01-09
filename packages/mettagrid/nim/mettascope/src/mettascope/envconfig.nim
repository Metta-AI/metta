import
  std/[json, os],
  boxy, silky, webby, windy,
  common, replays, panels

proc drawEnvironmentInfo*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  frame(frameId, contentPos, contentSize):
    text("Environment Info")
    button("Open Config"):
      let text =
        if replay.isNil or replay.mgConfig.isNil:
          "No replay config found."
        else:
          replay.mgConfig.pretty
      openTempTextFile("mg_config.json", text)
