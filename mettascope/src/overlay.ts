import * as Common from './common.js'
import { ctx, state, ui } from './common.js'
import { sendVisualSetLayer } from './replay.js'

const overlayFontId = 'plexSans'

function ensureObsOverlayMenu(): void {
  let menu = document.getElementById('obs-overlay-menu') as HTMLDivElement | null
  if (menu) {
    return
  }
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
  const tileSize = Common.TILE_SIZE

  const fontSize = 64
  const scale = (tileSize * 0.55) / fontSize

  for (let r = 0; r < g.height; r++) {
    for (let c = 0; c < g.width; c++) {
      const idx = r * g.width + c
      const value = g.values[idx]
      if (value === 0) continue // don't display 0 tensors.
      const text = String(value)

      // Align overlay with the step the agent was on when we got the snapshot.
      // this prevents a flicking when the agent is moving.
      const agentStep = state.visualGrid.step
      const agentLoc = agentObj.location.get(agentStep)
      const wx = agentLoc[0] + (c - halfW)
      const wy = agentLoc[1] + (r - halfH)

      const centerX = wx * tileSize
      const centerY = wy * tileSize

      ctx.save()
      ctx.translate(centerX, centerY)
      ctx.scale(scale, scale)
      const { width: textWidth, top, bottom } = ctx.measureTextBounds(overlayFontId, text)
      const textHeight = bottom - top
      ctx.translate(-textWidth / 2, -(top + textHeight / 2))
      ctx.drawText(overlayFontId, text, [1, 1, 1, 1])
      ctx.restore()
    }
  }
}
