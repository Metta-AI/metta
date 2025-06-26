import { Vec2f, Mat3f } from './vector_math.js'
import * as Common from './common.js'
import { ui, state, html, ctx, setFollowSelection } from './common.js'
import { fetchReplay, getAttr, initWebSocket, readFile, sendAction } from './replay.js'
import { focusFullMap, drawMap } from './worldmap.js'
import { drawTrace } from './traces.js'
import { drawMiniMap } from './minimap.js'
import { processActions, initActionButtons } from './actions.js'
import { initAgentTable, updateAgentTable } from './agentpanel.js'
import { localStorageSetNumber, onEvent, initHighDpiMode, find } from './htmlutils.js'
import { updateReadout, hideHoverPanel } from './hoverpanels.js'
import { initObjectMenu } from './objmenu.js'
import { drawTimeline, initTimeline, updateTimeline, onScrubberChange } from './timeline.js'
import { initDemoMode, startDemoMode, stopDemoMode, doDemoMode } from './demomode.js'


/** A flag to prevent multiple calls to requestAnimationFrame. */
let frameRequested = false

/** A function to safely request an animation frame. */
export function requestFrame() {
  if (!frameRequested) {
    frameRequested = true
    requestAnimationFrame((time) => {
      frameRequested = false
      onFrame()
    })
  }
}

/** Handles resize events. */
export function onResize() {
  // Adjust for high DPI displays.
  const dpr = window.devicePixelRatio || 1

  const screenWidth = window.innerWidth
  const screenHeight = window.innerHeight

  if (!state.showUi) {
    ui.mapPanel.x = 0
    ui.mapPanel.y = 0
    ui.mapPanel.width = screenWidth
    ui.mapPanel.height = screenHeight
    ui.miniMapPanel.height = 0
    ui.infoPanel.height = 0
    ui.tracePanel.width = 0
    ui.tracePanel.height = 0
    ui.timelinePanel.height = 0
    ui.agentPanel.height = 0
  } else {
    // Make sure traceSplit and infoSplit are not too small or too large.
    const a = 0.025
    ui.traceSplit = Math.max(a, Math.min(ui.traceSplit, 1 - a))
    ui.agentPanelSplit = Math.max(a, Math.min(ui.agentPanelSplit, 1 - a))

    ui.mapPanel.x = 0
    ui.mapPanel.y = Common.HEADER_HEIGHT
    ui.mapPanel.width = screenWidth
    let maxMapHeight = screenHeight - Common.HEADER_HEIGHT - Common.FOOTER_HEIGHT
    ui.mapPanel.height = Math.min(screenHeight * ui.traceSplit - Common.HEADER_HEIGHT, maxMapHeight)

    // Minimap goes in the bottom left corner of the mapPanel.
    if (state.replay != null) {
      const miniMapWidth = state.replay.map_size[0] * 2
      const miniMapHeight = state.replay.map_size[1] * 2
      ui.miniMapPanel.x = 0
      ui.miniMapPanel.y = ui.mapPanel.y + ui.mapPanel.height - miniMapHeight
      ui.miniMapPanel.width = miniMapWidth
      ui.miniMapPanel.height = miniMapHeight
    }

    ui.infoPanel.x = screenWidth - 400
    ui.infoPanel.y = ui.mapPanel.y + ui.mapPanel.height - 300
    ui.infoPanel.width = 400
    ui.infoPanel.height = 300

    // Trace panel is always on the bottom of the screen.
    if (state.showTraces) {
      ui.tracePanel.x = 0
      ui.tracePanel.y = ui.mapPanel.y + ui.mapPanel.height
      ui.tracePanel.width = screenWidth
      ui.tracePanel.height = screenHeight - ui.tracePanel.y - Common.FOOTER_HEIGHT

      html.actionButtons.style.top = ui.tracePanel.y - 148 + 'px'
    } else {
      ui.tracePanel.x = 0
      ui.tracePanel.y = 0
      ui.tracePanel.width = 0
      ui.tracePanel.height = 0
      // Have the map panel take up the trace panel's space.
      ui.mapPanel.height = screenHeight - ui.mapPanel.y - Common.FOOTER_HEIGHT
      ui.miniMapPanel.y = ui.mapPanel.y + ui.mapPanel.height - ui.miniMapPanel.height
      ui.infoPanel.y = ui.mapPanel.y + ui.mapPanel.height - 300
      html.actionButtons.style.top = ui.mapPanel.y + ui.mapPanel.height - 148 + 'px'
    }

    // Timeline panel is always on the bottom of the screen.
    ui.timelinePanel.x = 0
    ui.timelinePanel.y = screenHeight - 64 - 64
    ui.timelinePanel.width = screenWidth
    ui.timelinePanel.height = 64

    // Agent panel is always on the top of the screen.
    ui.agentPanel.x = 0
    ui.agentPanel.y = Common.HEADER_HEIGHT
    ui.agentPanel.width = screenWidth
    ui.agentPanel.height = ui.agentPanelSplit * screenHeight
  }

  ui.mapPanel.updateDiv()
  ui.miniMapPanel.updateDiv()
  ui.infoPanel.updateDiv()
  ui.tracePanel.updateDiv()
  ui.agentPanel.updateDiv()
  ui.timelinePanel.updateDiv()

  updateTimeline()

  // Redraw the square after resizing.
  requestFrame()
}

/** Shows all UI elements. */
function showUi() {
  find("#header").classList.remove('hidden')
  find("#footer").classList.remove('hidden')
  onResize()
}

/** Hides all UI elements. */
function hideUi() {
  find("#header").classList.add('hidden')
  find("#footer").classList.add('hidden')
  state.showMiniMap = false
  state.showInfo = false
  state.showTraces = false
  state.showAgentPanel = false
  state.showActionButtons = false
  onResize()
}

/** Handles mouse down events. */
onEvent('mousedown', 'body', () => {
  ui.lastMousePos = ui.mousePos
  ui.mouseDownPos = ui.mousePos
  ui.mouseClick = true

  if (Math.abs(ui.mousePos.y() - ui.tracePanel.y) < Common.SPLIT_DRAG_THRESHOLD) {
    ui.dragging = 'trace-panel'
  } else {
    ui.mouseDown = true
  }

  if (Math.abs(ui.mousePos.y() - (ui.agentPanel.y + ui.agentPanel.height)) < Common.SPLIT_DRAG_THRESHOLD) {
    ui.dragging = 'agent-panel'
  } else {
    ui.mouseDown = true
  }

  requestFrame()
})

/** Handles mouse up events. */
onEvent('mouseup', 'body', () => {
  ui.mouseUp = true
  ui.mouseDown = false
  ui.dragging = ''
  ui.dragHtml = null
  ui.dragOffset = new Vec2f(0, 0)
  ui.mainScrubberDown = false

  // Due to how we select objects on mouse-up (mouse-down is drag/pan),
  // we need to check for double-click on mouse-up as well.
  const currentTime = new Date().getTime()
  ui.mouseDoubleClick = currentTime - ui.lastClickTime < 300 // 300ms threshold for double-click
  ui.lastClickTime = currentTime

  requestFrame()
})

/** Handles mouse move events. */
onEvent('mousemove', 'body', (target: HTMLElement, e: Event) => {
  let event = e as MouseEvent
  ui.mousePos = new Vec2f(event.clientX, event.clientY)
  var target = event.target as HTMLElement
  while (target.id === '' && target.parentElement != null) {
    target = target.parentElement as HTMLElement
  }
  ui.mouseTargets = []
  let p = event.target as HTMLElement
  while (p != null) {
    if (p.id !== '') {
      ui.mouseTargets.push('#' + p.id)
    }
    for (const className of p.classList) {
      ui.mouseTargets.push('.' + className)
    }
    p = p.parentElement as HTMLElement
  }

  // If the mouse is close to a panel's edge, change the cursor.
  document.body.style.cursor = 'default'
  if (Math.abs(ui.mousePos.y() - ui.tracePanel.y) < Common.SPLIT_DRAG_THRESHOLD) {
    document.body.style.cursor = 'ns-resize'
  }
  if (
    state.showAgentPanel &&
    Math.abs(ui.mousePos.y() - (ui.agentPanel.y + ui.agentPanel.height)) < Common.SPLIT_DRAG_THRESHOLD
  ) {
    document.body.style.cursor = 'ns-resize'
  }

  // Drag the trace panel up or down.
  if (ui.dragging == 'trace-panel') {
    ui.traceSplit = ui.mousePos.y() / window.innerHeight
    localStorageSetNumber('traceSplit', ui.traceSplit)
    onResize()
  }

  if (ui.dragging == 'agent-panel') {
    ui.agentPanelSplit = (ui.mousePos.y() - ui.agentPanel.y) / window.innerHeight
    localStorageSetNumber('agentPanelSplit', ui.agentPanelSplit)
    onResize()
  }

  if (ui.dragHtml != null) {
    ui.dragHtml.style.left = ui.mousePos.x() - ui.dragOffset.x() + 'px'
    ui.dragHtml.style.top = ui.mousePos.y() - ui.dragOffset.y() + 'px'
  }

  if (ui.mainScrubberDown) {
    onScrubberChange(event)
  }

  if (!ui.mouseTargets.includes('#worldmap-panel') && !ui.mouseTargets.includes('.hover-panel')) {
    hideHoverPanel()
  }

  requestFrame()
})

/** Handles dragging draggable elements. */
onEvent('mousedown', '.draggable', (target: HTMLElement, e: Event) => {
  let event = e as MouseEvent
  ui.dragHtml = target
  let rect = target.getBoundingClientRect()
  ui.dragOffset = new Vec2f(event.clientX - rect.left, event.clientY - rect.top)
  ui.dragging = 'draggable'
  requestFrame()
})

/** Handles scroll events. */
onEvent('wheel', 'body', (target: HTMLElement, e: Event) => {
  let event = e as WheelEvent
  ui.scrollDelta = event.deltaY
  // Prevent pinch-to-zoom.
  event.preventDefault()
  requestFrame()
})

/** Handles the mouse moving outside the window. */
document.addEventListener('mouseout', function (e) {
  if (!e.relatedTarget) {
    hideHoverPanel()
    requestFrame()
  }
})

/** Handles the window losing focus. */
document.addEventListener('blur', function (e) {
  hideHoverPanel()
  requestFrame()
})

/** Updates all URL parameters without creating browser history entries. */
function updateUrlParams() {
  // Get the current URL parameters.
  const urlParams = new URLSearchParams(window.location.search)

  // Update the step when it's not zero.
  if (state.step !== 0) {
    urlParams.set('step', state.step.toString())
  } else {
    urlParams.delete('step')
  }

  // Handle the selected object.
  if (state.selectedGridObject !== null) {
    // Find the index of the selected object.
    const selectedObjectIndex = state.replay.grid_objects.indexOf(state.selectedGridObject)
    if (selectedObjectIndex !== -1) {
      urlParams.set('selectedObjectId', (selectedObjectIndex + 1).toString())
      // Remove map position parameters when an object is selected.
      urlParams.delete('mapPanX')
      urlParams.delete('mapPanY')
    }
  } else {
    // Include the map position.
    urlParams.set('mapPanX', Math.round(ui.mapPanel.panPos.x()).toString())
    urlParams.set('mapPanY', Math.round(ui.mapPanel.panPos.y()).toString())
    // Remove the selected object when there is no selection.
    urlParams.delete('selectedObjectId')
  }

  // Include the map zoom level.
  if (ui.mapPanel.zoomLevel != 1) {
    // Only include zoom to three decimal places.
    urlParams.set('mapZoom', ui.mapPanel.zoomLevel.toFixed(3))
  }

  // Handle the play state; only include it when true.
  if (state.isPlaying) {
    urlParams.set('play', 'true')
  } else {
    urlParams.delete('play')
  }

  // Replace the current state without creating a history entry.
  const newUrl = window.location.pathname + '?' + urlParams.toString()
  history.replaceState(null, '', newUrl)
}

/** Centralized function to update the step and handle all related updates. */
export function updateStep(newStep: number, skipScrubberUpdate = false) {
  // Update the step variable.
  state.step = newStep

  // Update the scrubber value (unless told to skip).
  if (!skipScrubberUpdate) {
    console.info('Scrubber value:', state.step)
    updateTimeline()
  }
  updateAgentTable()
  requestFrame()
}

/** Centralized function to select an object. */
export function updateSelection(object: any, setFollow = false) {
  state.selectedGridObject = object
  if (setFollow) {
    setFollowSelection(true)
  }
  console.info('Selected object:', state.selectedGridObject)
  updateAgentTable()
  requestFrame()
}

/** Handles key down events. */
onEvent('keydown', 'body', (target: HTMLElement, e: Event) => {
  let event = e as KeyboardEvent

  // Prevent keyboard events if we are focused on an text field.
  if (
    document.activeElement instanceof HTMLInputElement ||
    document.activeElement instanceof HTMLTextAreaElement
  ) {
    return
  }

  if (event.key == 'Escape') {
    updateSelection(null)
    setFollowSelection(false)
  }
  // '[' and ']' scrub forward and backward.
  if (event.key == '[') {
    setIsPlaying(false)
    updateStep(Math.max(state.step - 1, 0))
  }
  if (event.key == ']') {
    setIsPlaying(false)
    updateStep(Math.min(state.step + 1, state.replay.max_steps - 1))
  }
  // '<' and '>' control the playback speed.
  if (event.key == ',') {
    state.playbackSpeed = Math.max(state.playbackSpeed * 0.9, 0.01)
  }
  if (event.key == '.') {
    state.playbackSpeed = Math.min(state.playbackSpeed * 1.1, 1000)
  }
  // The space bar presses the play button.
  if (event.key == ' ') {
    setIsPlaying(!state.isPlaying)
  }
  // Make F2 toggle the UI.
  if (event.key == 'F2') {
    state.showUi = !state.showUi
    if (state.showUi) {
      showUi()
    } else {
      hideUi()
    }
    requestFrame()
  }

  processActions(event)

  requestFrame()
})

/** Draws a frame. */
export function onFrame() {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

  doDemoMode()

  ctx.clear()

  ui.mapPanel.updatePanAndZoom()
  ui.tracePanel.updatePanAndZoom()

  ctx.useMesh('map')
  drawMap(ui.mapPanel)

  if (state.showMiniMap) {
    ui.miniMapPanel.div.classList.remove('hidden')
    ctx.useMesh('mini-map')
    drawMiniMap(ui.miniMapPanel)
  } else {
    ui.miniMapPanel.div.classList.add('hidden')
  }

  if (state.showTraces) {
    ui.tracePanel.div.classList.remove('hidden')
    ctx.useMesh('trace')
    drawTrace(ui.tracePanel)
  } else {
    ui.tracePanel.div.classList.add('hidden')
  }

  ctx.useMesh('timeline')
  drawTimeline(ui.timelinePanel)

  if (state.showInfo) {
    ui.infoPanel.div.classList.remove('hidden')
    updateReadout()
  } else {
    ui.infoPanel.div.classList.add('hidden')
  }

  if (state.showActionButtons) {
    html.actionButtons.classList.remove('hidden')
  } else {
    html.actionButtons.classList.add('hidden')
  }

  if (state.showAgentPanel) {
    ui.agentPanel.div.classList.remove('hidden')
  } else {
    ui.agentPanel.div.classList.add('hidden')
  }

  ctx.flush()

  // Update URL parameters with the current state once per frame.
  updateUrlParams()

  if (state.isPlaying) {
    state.partialStep += state.playbackSpeed
    if (state.partialStep >= 1) {
      const nextStep = (state.step + Math.floor(state.partialStep)) % state.replay.max_steps
      state.partialStep -= Math.floor(state.partialStep)
      if (state.ws !== null) {
        state.ws.send(JSON.stringify({ type: 'advance' }))
      } else {
        updateStep(nextStep)
      }
    }
    requestFrame()
  }

  ui.mouseUp = false
  ui.mouseClick = false
  ui.mouseDoubleClick = false
}

/** Prevents default event handling. */
function preventDefaults(event: Event) {
  event.preventDefault()
  event.stopPropagation()
}

/** Handles file drop events. */
function handleDrop(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  const dt = event.dataTransfer
  if (dt && dt.files.length) {
    const file = dt.files[0]
    readFile(file)
  }
}

/** Parses URL parameters and modifies the map and trace panels accordingly. */
async function parseUrlParams() {
  const urlParams = new URLSearchParams(window.location.search)

  // Load the replay.
  const replayUrl = urlParams.get('replayUrl')
  const wsUrl = urlParams.get('wsUrl')
  if (replayUrl) {
    console.info('Loading replay from URL: ', replayUrl)
    await fetchReplay(replayUrl)
    focusFullMap(ui.mapPanel)
  } else if (wsUrl) {
    Common.showModal('info', 'Connecting to a websocket', 'Please wait a few seconds for the environment to load.')
    initWebSocket(wsUrl)
  } else {
    Common.showModal('info', 'Welcome to MettaScope', 'Please drop a replay file here to see the replay.')
  }

  if (state.replay !== null) {
    // Set the current step.
    if (urlParams.get('step') !== null) {
      const initialStep = parseInt(urlParams.get('step') || '0')
      console.info('Step via query parameter:', initialStep)
      updateStep(initialStep, false)
    }

    // Set the playing state.
    if (urlParams.get('play') !== null) {
      setIsPlaying(urlParams.get('play') === 'true')
      console.info('Playing state via query parameter:', state.isPlaying)
    }

    // Set the selected object.
    if (urlParams.get('selectedObjectId') !== null) {
      const selectedObjectId = parseInt(urlParams.get('selectedObjectId') || '-1') - 1
      if (selectedObjectId >= 0 && selectedObjectId < state.replay.grid_objects.length) {
        updateSelection(state.replay.grid_objects[selectedObjectId], true)
        ui.mapPanel.zoomLevel = Common.DEFAULT_ZOOM_LEVEL
        ui.tracePanel.zoomLevel = Common.DEFAULT_TRACE_ZOOM_LEVEL
        console.info('Selected object via query parameter:', state.selectedGridObject)
      } else {
        console.warn('Invalid selectedObjectId:', selectedObjectId)
      }
    }
  }

  // Set the map pan and zoom.
  if (urlParams.get('mapPanX') !== null && urlParams.get('mapPanY') !== null) {
    const mapPanX = parseInt(urlParams.get('mapPanX') || '0')
    const mapPanY = parseInt(urlParams.get('mapPanY') || '0')
    ui.mapPanel.panPos = new Vec2f(mapPanX, mapPanY)
  }
  if (urlParams.get('mapZoom') !== null) {
    ui.mapPanel.zoomLevel = parseFloat(urlParams.get('mapZoom') || '1')
  }

  if (urlParams.get('demo') !== null) {
    startDemoMode()
  }

  requestFrame()
}

/** Handles share button clicks. */
function onShareButtonClick() {
  // Copy the current URL to the clipboard.
  navigator.clipboard.writeText(window.location.href)
  // Show a toast notification.
  Common.showToast('URL copied to clipboard')
}

/** Sets the playing state and updates the play button icon. */
export function setIsPlaying(isPlaying: boolean) {
  state.isPlaying = isPlaying
  if (state.isPlaying) {
    html.playButton.setAttribute('src', 'data/ui/pause.png')
  } else {
    html.playButton.setAttribute('src', 'data/ui/play.png')
  }
  requestFrame()
}

/** Toggles the opacity of a button. */
function toggleOpacity(button: HTMLElement, show: boolean) {
  if (show) {
    button.classList.remove('transparent')
  } else {
    button.classList.add('transparent')
  }
}

/** Sets the playback speed and updates the speed buttons. */
function setPlaybackSpeed(speed: number) {
  state.playbackSpeed = speed
  // Update the speed buttons to show the current speed.
  for (let i = 0; i < html.speedButtons.length; i++) {
    toggleOpacity(html.speedButtons[i], Common.SPEEDS[i] <= speed)
  }
}

// Initial resize.
onResize()

// Disable pinch-to-zoom.
let meta = document.createElement('meta')
meta.name = 'viewport'
meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
document.head.appendChild(meta)

html.modal.classList.add('hidden')
html.toast.classList.add('hiding')
html.actionButtons.classList.add('hidden')
ui.infoPanel.div.classList.add('hidden')
ui.agentPanel.div.classList.add('hidden')

// Each panel has a div we use for event handling.
// But rendering happens below on the global canvas.
// We make the divs transparent to see through them.
ui.mapPanel.div.style.backgroundColor = 'rgba(0, 0, 0, 0.0)'
ui.tracePanel.div.style.backgroundColor = 'rgba(0, 0, 0, 0.0)'
ui.miniMapPanel.div.style.backgroundColor = 'rgba(0, 0, 0, 0.0)'
ui.timelinePanel.div.style.backgroundColor = 'rgba(0, 0, 0, 0.0)'

// Add an event listener to resize the canvas when the window is resized.
window.addEventListener('resize', onResize)
window.addEventListener('dragenter', preventDefaults, false)
window.addEventListener('dragleave', preventDefaults, false)
window.addEventListener('dragover', preventDefaults, false)
window.addEventListener('drop', handleDrop, false)

// Header area
onEvent('click', '#share-button', () => {
  onShareButtonClick()
})
onEvent('click', '#help-button', () => {
  window.open('https://github.com/Metta-AI/metta/blob/main/mettascope/README.md', '_blank')
})

onEvent('click', '#rewind-to-start', () => {
  setIsPlaying(false)
  updateStep(0)
})
onEvent('click', '#step-back', () => {
  setIsPlaying(false)
  updateStep(Math.max(state.step - 1, 0))
})
onEvent('click', '#play', () => {
  setIsPlaying(!state.isPlaying)
})
onEvent('click', '#step-forward', () => {
  setIsPlaying(false)
  if (state.ws !== null) {
    state.ws.send(JSON.stringify({ type: 'advance' }))
  } else {
    updateStep(Math.min(state.step + 1, state.replay.max_steps - 1))
  }
})
onEvent('click', '#rewind-to-end', () => {
  setIsPlaying(false)
  updateStep(state.replay.max_steps - 1)
})
onEvent('click', '#demo-mode-toggle', () => {
  if (state.demoMode) {
    stopDemoMode()
  } else {
    startDemoMode()
  }
  toggleOpacity(html.demoModeToggle, state.demoMode)
  requestFrame()
})
toggleOpacity(html.demoModeToggle, state.demoMode)

onEvent('click', '#full-screen-toggle', () => {
  state.fullScreen = !state.fullScreen
  if (state.fullScreen) {
    document.documentElement.requestFullscreen()
  } else {
    document.exitFullscreen()
  }
  toggleOpacity(html.fullScreenToggle, state.fullScreen)
})
toggleOpacity(html.fullScreenToggle, state.fullScreen)

// Speed buttons
for (let i = 0; i < html.speedButtons.length; i++) {
  html.speedButtons[i].addEventListener('click', () => setPlaybackSpeed(Common.SPEEDS[i]))
}

onEvent('click', '#resources-toggle', () => {
  state.showResources = !state.showResources
  localStorage.setItem('showResources', state.showResources.toString())
  toggleOpacity(html.resourcesToggle, state.showResources)
  requestFrame()
})
if (localStorage.hasOwnProperty('showResources')) {
  state.showResources = localStorage.getItem('showResources') === 'true'
}
toggleOpacity(html.resourcesToggle, state.showResources)

// Toggle follow selection state.
onEvent('click', '#focus-toggle', () => {
  setFollowSelection(!state.followSelection)
})
toggleOpacity(html.focusToggle, state.followSelection)

onEvent('click', '#grid-toggle', () => {
  state.showGrid = !state.showGrid
  localStorage.setItem('showGrid', state.showGrid.toString())
  toggleOpacity(html.gridToggle, state.showGrid)
  requestFrame()
})
if (localStorage.hasOwnProperty('showGrid')) {
  state.showGrid = localStorage.getItem('showGrid') === 'true'
}
toggleOpacity(html.gridToggle, state.showGrid)

onEvent('click', '#visual-range-toggle', () => {
  state.showVisualRanges = !state.showVisualRanges
  localStorage.setItem('showVisualRanges', state.showVisualRanges.toString())
  toggleOpacity(html.visualRangeToggle, state.showVisualRanges)
  requestFrame()
})
if (localStorage.hasOwnProperty('showVisualRanges')) {
  state.showVisualRanges = localStorage.getItem('showVisualRanges') === 'true'
}
toggleOpacity(html.visualRangeToggle, state.showVisualRanges)

onEvent('click', '#fog-of-war-toggle', () => {
  state.showFogOfWar = !state.showFogOfWar
  localStorage.setItem('showFogOfWar', state.showFogOfWar.toString())
  toggleOpacity(html.fogOfWarToggle, state.showFogOfWar)
  requestFrame()
})
if (localStorage.hasOwnProperty('showFogOfWar')) {
  state.showFogOfWar = localStorage.getItem('showFogOfWar') === 'true'
}
toggleOpacity(html.fogOfWarToggle, state.showFogOfWar)

onEvent('click', '#minimap-toggle', () => {
  state.showMiniMap = !state.showMiniMap
  localStorage.setItem('showMiniMap', state.showMiniMap.toString())
  toggleOpacity(html.minimapToggle, state.showMiniMap)
  requestFrame()
})
if (localStorage.hasOwnProperty('showMiniMap')) {
  state.showMiniMap = localStorage.getItem('showMiniMap') === 'true'
}
toggleOpacity(html.minimapToggle, state.showMiniMap)

onEvent('click', '#controls-toggle', () => {
  state.showActionButtons = !state.showActionButtons
  localStorage.setItem('showActionButtons', state.showActionButtons.toString())
  toggleOpacity(html.controlsToggle, state.showActionButtons)
  requestFrame()
})
if (localStorage.hasOwnProperty('showActionButtons')) {
  state.showActionButtons = localStorage.getItem('showActionButtons') === 'true'
}
toggleOpacity(html.controlsToggle, state.showActionButtons)

onEvent('click', '#info-toggle', () => {
  state.showInfo = !state.showInfo
  localStorage.setItem('showInfo', state.showInfo.toString())
  toggleOpacity(html.infoToggle, state.showInfo)
  requestFrame()
})
if (localStorage.hasOwnProperty('showInfo')) {
  state.showInfo = localStorage.getItem('showInfo') === 'true'
}
toggleOpacity(html.infoToggle, state.showInfo)

onEvent('click', '#agent-panel-toggle', () => {
  state.showAgentPanel = !state.showAgentPanel
  localStorage.setItem('showAgentPanel', state.showAgentPanel.toString())
  toggleOpacity(html.agentPanelToggle, state.showAgentPanel)
  requestFrame()
})
if (localStorage.hasOwnProperty('showAgentPanel')) {
  state.showAgentPanel = localStorage.getItem('showAgentPanel') === 'true'
}
toggleOpacity(html.agentPanelToggle, state.showAgentPanel)

onEvent('click', '#traces-toggle', () => {
  state.showTraces = !state.showTraces
  localStorage.setItem('showTraces', state.showTraces.toString())
  toggleOpacity(html.tracesToggle, state.showTraces)
  onResize()
  requestFrame()
})
if (localStorage.hasOwnProperty('showTraces')) {
  state.showTraces = localStorage.getItem('showTraces') === 'true'
}
toggleOpacity(html.tracesToggle, state.showTraces)

onEvent('click', '#info-panel .close', () => {
  state.showInfo = false
  localStorage.setItem('showInfo', state.showInfo.toString())
  toggleOpacity(html.infoToggle, state.showInfo)
  requestFrame()
})

onEvent('click', '#minimap-panel .close', () => {
  state.showMiniMap = false
  localStorage.setItem('showMiniMap', state.showMiniMap.toString())
  toggleOpacity(html.minimapToggle, state.showMiniMap)
  requestFrame()
})

onEvent('click', '#agent-panel .close', () => {
  state.showAgentPanel = false
  localStorage.setItem('showAgentPanel', state.showAgentPanel.toString())
  toggleOpacity(html.agentPanelToggle, state.showAgentPanel)
  requestFrame()
})

onEvent('click', '#trace-panel .close', () => {
  state.showTraces = false
  localStorage.setItem('showTraces', state.showTraces.toString())
  toggleOpacity(html.tracesToggle, state.showTraces)
  onResize()
  requestFrame()
})

onEvent('click', '#action-buttons .close', () => {
  state.showActionButtons = false
  localStorage.setItem('showActionButtons', state.showActionButtons.toString())
  toggleOpacity(html.controlsToggle, state.showActionButtons)
  requestFrame()
})

initHighDpiMode()
initActionButtons()
initAgentTable()
initObjectMenu()
initTimeline()
initDemoMode()

window.addEventListener('load', async () => {

  // Use a local atlas texture.
  const atlasImageUrl = 'dist/atlas.png'
  const atlasJsonUrl = 'dist/atlas.json'

  const success = await ctx.init(atlasJsonUrl, atlasImageUrl)
  if (!success) {
    Common.showModal('error', 'Initialization failed', 'Please check the console for more information.')
    return
  } else {
    console.info('Context3d initialized successfully.')
  }

  // Match the DPI scale between the HTML and the GPU.
  ui.dpr = ctx.dpr

  await parseUrlParams()
  setPlaybackSpeed(0.1)

  requestFrame()
})
