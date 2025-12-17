import
  std/[strutils, json],
  silky, chroma, vmath, windy,
  common

const
  HeaderColor = parseHtmlColor("#273646").rgbx

proc drawHeader*() =
  ribbon(sk.pos, vec2(sk.size.x, 64), HeaderColor):
    image("ui/logo")
    sk.advance(vec2(8, 2))
    var title = "Mettascope"

    if not common.replay.isNil:
      if common.replay.mgConfig != nil and common.replay.mgConfig.contains("label"):
        let node = common.replay.mgConfig["label"]
        if node.kind == JString:
          title = node.getStr
      if title == "Mettascope" and common.replay.fileName.len > 0:
        title = common.replay.fileName
    h1text(title)

    sk.at = sk.pos + vec2(sk.size.x - 100, 16)
    iconButton("ui/help"):
      openUrl("https://github.com/Metta-AI/metta/blob/main/packages/mettagrid/nim/mettascope/README.md")
    iconButton("ui/share"):
      let baseUrl = "https://metta-ai.github.io/metta/mettascope/mettascope.html?replay="
      let url = baseUrl & commandLineReplay
      openUrl(url)
