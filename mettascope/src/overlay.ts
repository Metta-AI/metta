import * as Common from './common.js'
import { ctx, state, ui } from './common.js'
import { sendVisualSetLayer } from './replay.js'

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

    const status = document.createElement('div')
    status.id = 'obs-overlay-agent-status'
    status.style.marginBottom = '6px'
    menu.appendChild(status)

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
  // Setup the overlay menu and make sure it matches the current state.
  ensureObsOverlayMenu()
  const menu = document.getElementById('obs-overlay-menu') as HTMLDivElement
  const list = document.getElementById('obs-overlay-menu-list') as HTMLDivElement
  const status = document.getElementById('obs-overlay-agent-status') as HTMLDivElement
  menu.style.display = state.ws !== null && state.showObsOverlay && state.visualLayers.length > 0 ? 'block' : 'none'

  // Update agent selection status line.
  const sel = state.selectedGridObject
  if (sel && sel.isAgent) {
    status.textContent = `Selected agent: #${sel.agentId}`
  } else {
    status.textContent = 'No agent selected'
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

/** Draws the observation tensor overlay around the selected agent (play mode only). */
export function drawObservationOverlay(): void {
  if (state.ws === null || !state.showObsOverlay || state.visualGrid === null) {
    return
  }
  const g = state.visualGrid

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
  // Draw per‑tile numeric text (0–255) using the atlas font.
  const tileSize = Common.TILE_SIZE
  const fontId = 'plexSans'
  if (!ctx.atlasData || !(ctx.atlasData as any).fonts || !(ctx.atlasData as any).fonts[fontId]) {
    return
  }
  const font = (ctx.atlasData as any).fonts[fontId]
  // Scale text to occupy about half of the tile height; fall back to font size 64 if metadata not ready.
  const baseLineHeight: number = (ctx.atlasData as any)?.fonts?.[fontId]?.lineHeight ?? 64
  const scale = (tileSize * 0.55) / baseLineHeight


  for (let r = 0; r < g.height; r++) {
    for (let c = 0; c < g.width; c++) {
      const idx = r * g.width + c
      if (idx < 0 || idx >= g.values.length) continue
      const rawVal = Number((g.values as any)[idx] ?? 0)
      const value = Math.max(0, Math.min(255, Math.round(rawVal)))
      if (value === 0) continue
      const text = String(value)

      const wx = ax + (c - halfW)
      const wy = ay + (r - halfH)
      if (wx < 0 || wy < 0 || wx >= state.replay.mapSize[0] || wy >= state.replay.mapSize[1]) continue

      const alpha = g.valueRange && g.valueRange.max !== g.valueRange.min
        ? Math.min(1, Math.max(0, (rawVal - g.valueRange.min) / (g.valueRange.max - g.valueRange.min)))
        : 1

      const centerX = wx * tileSize
      const centerY = wy * tileSize
      const a = Math.max(0, Math.min(1, 0.5 + 0.5 * alpha))

      ctx.save()
      ctx.translate(centerX, centerY)
      ctx.scale(scale, scale)
      // Center text around the tile center using measured bounds.
      const { width: textWidth, top, bottom } = ctx.measureTextBounds(fontId, text)
      const textHeight = bottom - top
      // Move origin so the visual bounds are centered.
      ctx.translate(-textWidth / 2, -(top + textHeight / 2))
      ctx.drawText(fontId, text, [1, 1, 1, a])
      ctx.restore()
    }
  }
}

