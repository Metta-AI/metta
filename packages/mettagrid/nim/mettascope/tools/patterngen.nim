import tables,pixie

# This file reads the blob7x7.png and computes the adjacency tile mapping.

let blob7x7 = readImage("tools/blob7x7plain.png")

let ts = 64

var patternToTile = newSeq[int](256)
for y in 0 ..< 7:
  for x in 0 ..< 7:
    # First we look at each tile and figure out which neighbors it should have.
    let tileNum = x + y * 7
    echo "looking at tile: ", tileNum
    let nw = blob7x7[x * ts, y * ts].r == 255
    echo "  nw: ", nw
    let n = blob7x7[x * ts + ts div 2, y * ts].r == 255
    echo "  n: ", n
    let ne = blob7x7[x * ts + ts - 1, y * ts].r == 255
    echo "  ne: ", ne
    let e = blob7x7[x * ts + ts - 1, y * ts + ts div 2].r == 255
    echo "  e: ", e
    let se = blob7x7[x * ts + ts - 1, y * ts + ts - 1].r == 255
    echo "  se: ", se
    let s = blob7x7[x * ts + ts div 2, y * ts + ts - 1].r == 255
    echo "  s: ", s
    let sw = blob7x7[x * ts, y * ts + ts - 1].r == 255
    echo "  sw: ", sw
    let w = blob7x7[x * ts, y * ts + ts div 2].r == 255
    echo "  w: ", w

    let pattern = (
      nw.int * 1 +
      n.int * 2 +
      ne.int * 4 +
      e.int * 8 +
      se.int * 16 +
      s.int * 32 +
      sw.int * 64 +
      w.int * 128
    )
    echo "  pattern: ", pattern
    patternToTile[pattern] = tileNum

for pattern in 0 ..< 256:
  # We map dump power of 8 patterns to blob 47 patterns that ignore the corners
  # when edges are connected.
  echo "pattern: ", pattern
  var
    nw = (pattern and 1) == 1
    n = (pattern and 2) == 2
    ne = (pattern and 4) == 4
    e = (pattern and 8) == 8
    se = (pattern and 16) == 16
    s = (pattern and 32) == 32
    sw = (pattern and 64) == 64
    w = (pattern and 128) == 128

  # Corners don't matter.
  if n:
    nw = true
    ne = true
  if e:
    se = true
    ne = true
  if s:
    sw = true
    se = true
  if w:
    nw = true
    sw = true
  let newPattern = (
    nw.int * 1 +
    n.int * 2 +
    ne.int * 4 +
    e.int * 8 +
    se.int * 16 +
    s.int * 32 +
    sw.int * 64 +
    w.int * 128
  )
  echo "  maps to: ", newPattern
  patternToTile[pattern] = patternToTile[newPattern]
  echo "  tile: ", patternToTile[pattern]

echo "patternToTile: ", patternToTile
