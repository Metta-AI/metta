import std/[tables, algorithm, strutils],
  pixie, vmath, bumpy

# Generate 8 neighboring tiles template.

proc sorted(arr: seq[string]): seq[string] =
  var arr2 = arr
  arr2.sort()
  return arr2

let dirNames = [
  "n",
  "ne",
  "e",
  "se",
  "s",
  "sw",
  "w",
  "nw",
]

let dirDeltas = [
  (0, -1),
  (1, -1),
  (1, 0),
  (1, 1),
  (0, 1),
  (-1, 1),
  (-1, 0),
  (-1, -1)
]

let size = 64

let wTiles = newTable[string, Image]()
wTiles["n_ne_e_se_s_sw_w_nw"] = readImage("data/objects/wall.fill.png")
wTiles["zero"] = readImage("data/objects/floor1.png")

wTiles["e"] = readImage("data/objects/wall.e.png")
wTiles["n"] = readImage("data/objects/wall.n.png")
wTiles["n_e"] = readImage("data/objects/wall.ne.png")
wTiles["n_s"] = readImage("data/objects/wall.ns.png")
wTiles["n_s_e"] = readImage("data/objects/wall.nse.png")
wTiles["n_w"] = readImage("data/objects/wall.nw.png")
wTiles["n_w_e"] = readImage("data/objects/wall.nwe.png")
wTiles["n_w_s"] = readImage("data/objects/wall.nws.png")
wTiles["n_w_s_e"] = readImage("data/objects/wall.nwse.png")
wTiles["s"] = readImage("data/objects/wall.s.png")
wTiles["s_e"] = readImage("data/objects/wall.se.png")
wTiles["w"] = readImage("data/objects/wall.w.png")
wTiles["w_e"] = readImage("data/objects/wall.we.png")
wTiles["w_s"] = readImage("data/objects/wall.ws.png")
wTiles["w_s_e"] = readImage("data/objects/wall.wse.png")
for k in wTiles.keys:
  let k2 = k.split("_").sorted().join("_")
  echo k, " -> ", k2
  if k2 != k:
    wTiles[k2] = wTiles[k]
    wTiles.del(k)

let mainImage = newImage(16 * size, 16 * size)
#mainImage.fill(rgba(255, 255, 255, 255))
let mainCtx = newContext(mainImage)

for bitPattern in 0 ..< 256:
  let x = bitPattern mod 16
  let y = bitPattern div 16
  var nameParts = newSeq[string]()
  var forWayParts = newSeq[string]()
  for i in 0 ..< 8:
    if (bitPattern and (1 shl i)) != 0:
      nameParts.add(dirNames[i])
      if dirNames[i].len == 1:
        forWayParts.add(dirNames[i])
  if nameParts.len == 0:
    #nameParts = @["n", "e", "s", "w", "ne", "nw", "se", "sw"]
    nameParts.add("zero")
  if forWayParts.len == 0:
    forWayParts.add("zero")
    #forWayParts = @["n", "e", "s", "w"]
  let name = nameParts.sorted().join("_")
  let forWayName = forWayParts.sorted().join("_")
  echo x, " ", y, " ", name, " ", forWayName

  # If tile is one of the wTiles use it instead of drawing a square.
  var image: Image
  # if forWayName in wTiles:
  #   image = wTiles[forWayName].copy()
  # else:
  #   image = newImage(size, size)


  image = newImage(size, size)

  let ctx = newContext(image)

  # # Erase the edges:
  # ctx.fillStyle = color(0, 0, 0, 1)
  # if "ne" in nameParts:
  #   ctx.fillRect(48, 0, 16, 16)
  # if "nw" in nameParts:
  #   ctx.fillRect(0, 0, 16, 16)
  # if "se" in nameParts:
  #   ctx.fillRect(48, 48, 16, 16)
  # if "sw" in nameParts:
  #   ctx.fillRect(0, 48, 16, 16)

  ctx.fillStyle = color(1, 1, 1, 1)
  # Middle
  # ctx.beginPath()
  # ctx.rect(16, 16, 32, 32)
  # ctx.closePath()
  # ctx.fill()

  ctx.beginPath()
  for i in 0 ..< 8:
    if (bitPattern and (1 shl i)) != 0:
      # Draw a square at this direction.
      let delta = dirDeltas[i]
      let x = 32 + delta[0] * 48
      let y = 32 + delta[1] * 48
      ctx.rect(x.float32 - 32, y.float32 - 32, 64, 64)
  ctx.closePath()
  ctx.fill()


  # let pos = vec2(
  #   x.float32 * size.float32 * 2 + 32,
  #   y.float32 * size.float32 * 2 + 32
  # )

  let pos = vec2(
    x.float32 * size.float32,
    y.float32 * size.float32
  )
  mainImage.draw(image, translate(pos))

  # # Draw the bleeding borders:
  # # top
  # let topImg = image.subImage(0, 0, 64, 1)
  # for y in 0 ..< 32:
  #   mainImage.draw(topImg, translate(vec2(
  #     pos.x,
  #     pos.y - 1 - y.float32
  #   )))
  # # bottom
  # let bottomImg = image.subImage(0, 63, 64, 1)
  # for y in 0 ..< 32:
  #   mainImage.draw(bottomImg, translate(vec2(
  #     pos.x,
  #     pos.y + 64.float32 + y.float32
  #   )))
  # # left
  # let leftImg = image.subImage(0, 0, 1, 64)
  # for x in 0 ..< 32:
  #   mainImage.draw(leftImg, translate(vec2(
  #     pos.x - 1 - x.float32,
  #     pos.y
  #   )))
  # # right
  # let rightImg = image.subImage(63, 0, 1, 64)
  # for x in 0 ..< 32:
  #   mainImage.draw(rightImg, translate(vec2(
  #     pos.x + 64.float32 + x.float32,
  #     pos.y
  #   )))
  # # top left corner
  # mainCtx.fillStyle = image[0, 0]
  # mainCtx.beginPath()
  # mainCtx.rect(pos.x - 32, pos.y - 32, 32, 32)
  # mainCtx.closePath()
  # mainCtx.fill()
  # # top right corner
  # mainCtx.fillStyle = image[63, 0]
  # mainCtx.beginPath()
  # mainCtx.rect(pos.x + 64, pos.y - 32, 32, 32)
  # mainCtx.closePath()
  # mainCtx.fill()
  # # bottom left corner
  # mainCtx.fillStyle = image[0, 63]
  # mainCtx.beginPath()
  # mainCtx.rect(pos.x - 32, pos.y + 64, 32, 32)
  # mainCtx.closePath()
  # mainCtx.fill()
  # # bottom right corner
  # mainCtx.fillStyle = image[63, 63]
  # mainCtx.beginPath()
  # mainCtx.rect(pos.x + 64, pos.y + 64, 32, 32)
  # mainCtx.closePath()
  # mainCtx.fill()



mainImage.writeFile("tools/tilepuzzle.png")
