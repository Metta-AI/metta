import
  boxy, chroma,
  common, panels, replays, worldmap

proc drawMinimap*(panel: Panel) =
  let box = IRect(x: 0, y: 0, w: panel.rect.w, h: panel.rect.h)

bxy.drawRect(
  rect = box.rect,
  color = color(1, 0, 0, 1.0)
)

proc drawWorldMini*() =
  let wallTypeId = replay.typeNames.find("wall")
  let agentTypeId = replay.typeNames.find("agent")

  # Floor
  bxy.drawRect(rect(0, 0, replay.mapSize[0].float32 - 0.5,
      replay.mapSize[1].float32 - 0.5),
      color(0.906, 0.831, 0.718, 1))

  # Walls
  for obj in replay.objects:
    if obj.typeId == agentTypeId:
      continue
    let color =
      if obj.typeId == wallTypeId:
        color(0.380, 0.341, 0.294, 1)
      else:
        color(1, 1, 1, 1)

    let loc = obj.location.at(step).xy
    bxy.drawRect(rect(loc.x.float32 - 0.5, loc.y.float32 - 0.5, 1, 1), color)

  # Agents
  for obj in replay.objects:
    if obj.typeId != agentTypeId:
      continue

    let loc = obj.location.at(step).xy
    bxy.drawImage("minimapPip", rect(loc.x.float32 - 0.5, loc.y.float32 - 0.5,
        1, 1), agentColor(obj.agentId))

  # Overlays
  if settings.showVisualRange:
    drawVisualRanges()
  elif settings.showFogOfWar:
    drawFogOfWar()
