import * as Common from './common.js'
import { ctx, state, ui } from './common.js'
import { sendVisualSetLayer } from './replay.js'

/** Ensures the observation overlay menu exists in the DOM. */
function ensureObsOverlayMenu(): void {
  let menu = document.getElementById('obs-overlay-menu') as HTMLDivElement | null
  if (!menu) {
    menu = document.createElement('div')
    menu.id = 'obs-overlay-menu'
    menu.style.position = 'absolute'
    menu.style.top = `${ui.mapPanel.y + 8}px`
    menu.style.left = `${ui.mapPanel.x + 8}px`
    menu.style.zIndex = '5'
    menu.style.padding = '8px'
    menu.style.background = 'rgba(0,0,0,0.6)'
    menu.style.borderRadius = '8px'
    menu.style.color = '#fff'
    menu.style.fontFamily = 'sans-serif'
    menu.style.fontSize = '12px'
    menu.style.maxHeight = '200px'
    menu.style.overflowY = 'auto'
    menu.style.pointerEvents = 'auto'
    menu.style.display = 'none'

    const title = document.createElement('div')
    title.textContent = 'Observation Layers'
    title.style.fontWeight = 'bold'
    title.style.marginBottom = '6px'
    menu.appendChild(title)

    const list = document.createElement('div')
    list.id = 'obs-overlay-menu-list'
    menu.appendChild(list)

    // Prevent map interactions while interacting with the menu, but allow native scrolling.
    menu.addEventListener('wheel', (e) => {
      e.stopPropagation()
    })
    menu.addEventListener('pointerdown', (e) => {
      e.stopPropagation()
    })
    menu.addEventListener('mousedown', (e) => {
      e.stopPropagation()
    })
    menu.addEventListener('touchstart', (e) => {
      e.stopPropagation()
    })

    document.body.appendChild(menu)
  }
}

/** Updates the observation overlay menu items and selection state. */
export function updateObsOverlayMenu(): void {
  if (typeof document === 'undefined') {
    return
  }
  ensureObsOverlayMenu()
  const list = document.getElementById('obs-overlay-menu-list') as HTMLDivElement | null
  const menu = document.getElementById('obs-overlay-menu') as HTMLDivElement | null
  if (!list || !menu) {
    return
  }
  list.innerHTML = ''
  for (const layer of state.visualLayers) {
    const item = document.createElement('div')
    item.textContent = layer.name
    item.style.cursor = 'pointer'
    item.style.padding = '4px 6px'
    item.style.borderRadius = '4px'
    item.style.whiteSpace = 'nowrap'
    item.style.userSelect = 'none'
    if (state.activeVisualLayerId === layer.id) {
      item.style.background = 'rgba(255,255,255,0.15)'
      item.style.fontWeight = 'bold'
    }
    item.onclick = () => {
      state.activeVisualLayerId = layer.id
      sendVisualSetLayer(layer.id)
      updateObsOverlayMenu()
    }
    list.appendChild(item)
  }
  // Position near the map panel in case of resize.
  menu.style.top = `${ui.mapPanel.y + 8}px`
  menu.style.left = `${ui.mapPanel.x + 8}px`
}

/** Shows or hides the observation overlay menu based on state. */
export function setObsOverlayMenuVisibility(): void {
  if (typeof document === 'undefined') {
    return
  }
  const menu = document.getElementById('obs-overlay-menu') as HTMLDivElement | null
  if (!menu) {
    ensureObsOverlayMenu()
  }
  const menu2 = document.getElementById('obs-overlay-menu') as HTMLDivElement | null
  if (!menu2) {
    return
  }
  const shouldShow = state.ws !== null && state.showObsOverlay && state.visualLayers.length > 0
  menu2.style.display = shouldShow ? 'block' : 'none'
  if (shouldShow) {
    updateObsOverlayMenu()
  }
}

/** Draws the observation tensor overlay around the selected agent (play mode only). */
export function drawObservationOverlay(): void {
  if (state.ws === null || !state.showObsOverlay || state.visualGrid === null) {
    return
  }
  const g = state.visualGrid
  if (g.values.length < g.width * g.height) {
    return
  }
  const aid = state.activeVisualAgentId
  if (aid == null || aid < 0 || aid >= state.replay.agents.length) {
    return
  }
  const agentObj = state.replay.agents[aid]
  const loc = agentObj.location.get()
  const ax = loc[0]
  const ay = loc[1]
  const halfW = Math.floor(g.width / 2)
  const halfH = Math.floor(g.height / 2)
  ctx.save()
  // Draw per‑tile pips (dots) representing small integer values from 1–9.
  // The pips are placed on a 3×3 grid inside each tile and capped at 9.
  const tileSize = Common.TILE_SIZE
  const pipGrid = 3
  const inset = tileSize * 0.18
  const cell = (tileSize - inset * 2) / (pipGrid - 1)
  const pipSize = Math.max(2, Math.floor(tileSize * 0.08))
  // Precompute relative offsets for the 3×3 pip grid within a tile centered at (0,0).
  const offsets: [number, number][] = []
  for (let gy = 0; gy < pipGrid; gy++) {
    for (let gx = 0; gx < pipGrid; gx++) {
      const ox = -tileSize / 2 + inset + gx * cell
      const oy = -tileSize / 2 + inset + gy * cell
      offsets.push([ox, oy])
    }
  }
  for (let r = 0; r < g.height; r++) {
    for (let c = 0; c < g.width; c++) {
      const idx = r * g.width + c
      if (idx < 0 || idx >= g.values.length) continue
      const rawVal = (g.values as any)[idx] || 0
      if (rawVal === 0) continue
      const count = Math.max(0, Math.min(9, Number(rawVal)))
      const wx = ax + (c - halfW)
      const wy = ay + (r - halfH)
      if (wx < 0 || wy < 0 || wx >= state.replay.mapSize[0] || wy >= state.replay.mapSize[1]) continue
      const alpha = g.valueRange && g.valueRange.max !== g.valueRange.min
        ? Math.min(1, Math.max(0, (rawVal - g.valueRange.min) / (g.valueRange.max - g.valueRange.min)))
        : 1
      // Draw up to `count` pips inside this tile.
      const centerX = wx * tileSize
      const centerY = wy * tileSize
      for (let i = 0; i < count; i++) {
        const [ox, oy] = offsets[i]
        const px = centerX + ox
        const py = centerY + oy
        // Try to use a round sprite if available; otherwise draw a small square.
        if (ctx.hasImage('minimapPip.png')) {
          ctx.drawSprite('minimapPip.png', px, py, [1, 1, 1, 0.5 + 0.5 * alpha], pipSize / 2)
        } else {
          ctx.drawSolidRect(px - pipSize / 2, py - pipSize / 2, pipSize, pipSize, [1, 1, 1, 0.5 + 0.5 * alpha])
        }
      }
    }
  }
  ctx.restore()
}

