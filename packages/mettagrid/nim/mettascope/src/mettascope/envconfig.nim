import
  std/[json, os],
  boxy, silky, webby, windy,
  common, replays, panels

# var envConfig: Node

# find "/UI/EnvironmentInfo":
#   onLoad:
#     envConfig = find("/UI/EnvironmentInfo").copy()
#     envConfig.position = vec2(0, 0)

# find "/UI/Main/**/EnvironmentInfo/OpenConfig":
#   onClick:
#     let text =
#       if replay.isNil or replay.mgConfig.isNil:
#         "No replay config found."
#       else:
#         replay.mgConfig.pretty
#     openTempTextFile("mg_config.json", text)

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

proc updateEnvConfig*() =
  ## Updates the environment config panel to
  #environmentInfoPanel.node.addChild(envConfig)
  discard
