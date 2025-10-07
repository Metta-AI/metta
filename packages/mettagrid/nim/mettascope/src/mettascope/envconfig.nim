import
  std/[json, os],
  boxy, chroma, fidget2, jsony,
  common, panels, replays

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
    if not existsDir("tmp"):
      createDir("tmp")
    writeFile("tmp/mg_config.json", text)
    when defined(windows):
      discard execShellCmd("notepad tmp/mg_config.json")
    elif defined(macosx):
      discard execShellCmd("open -a TextEdit tmp/mg_config.json")
    else:
      discard execShellCmd("xdg-open tmp/mg_config.json")

proc updateEnvConfig*() =
  ## Updates the environment config panel to
  environmentInfoPanel.node.addChild(envConfig)
