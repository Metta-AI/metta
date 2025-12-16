import
  std/[json, os],
  boxy, fidget2, webby, windy,
  common, replays

var envConfig: Node

find "/UI/EnvironmentInfo":
  onLoad:
    envConfig = find("/UI/EnvironmentInfo").copy()
    envConfig.position = vec2(0, 0)

find "/UI/Main/**/EnvironmentInfo/OpenConfig":
  onClick:
    let text =
      if replay.isNil or replay.mgConfig.isNil:
        "No replay config found."
      else:
        replay.mgConfig.pretty
    openTempTextFile("mg_config.json", text)

proc updateEnvConfig*() =
  ## Updates the environment config panel to
  environmentInfoPanel.node.addChild(envConfig)
