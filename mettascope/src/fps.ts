/**
 * Self-contained FPS counter module that injects itself into the DOM
 * Similar to the glyph editor implementation
 */

import { onEvent } from './htmlutils.js'
import { requestFrame } from './main.js'

// FPS counter state
const fpsState = {
  fps: 0,
  frameCount: 0,
  lastTime: performance.now(),
  lastFpsUpdate: 0,
  fpsHistory: [] as number[],
  maxHistoryLength: 60,

  // Display options
  showFps: true,
  showAdvancedMetrics: false,

  // Performance metrics
  minFps: Infinity,
  maxFps: 0,
  avgFps: 0,
  frameTime: 0,

  // WebGL stats (will be populated by Context3d)
  drawCalls: 0,
  vertices: 0,
  triangles: 0,
}

// Load saved preferences
function loadFpsPreferences() {
  const showFps = localStorage.getItem('showFps')
  if (showFps !== null) {
    fpsState.showFps = showFps === 'true'
  }

  const showAdvanced = localStorage.getItem('showAdvancedFpsMetrics')
  if (showAdvanced !== null) {
    fpsState.showAdvancedMetrics = showAdvanced === 'true'
  }
}

// Save preferences
function saveFpsPreferences() {
  localStorage.setItem('showFps', fpsState.showFps.toString())
  localStorage.setItem('showAdvancedFpsMetrics', fpsState.showAdvancedMetrics.toString())
}

// Update FPS calculation
export function updateFps(webglStats?: { drawCalls: number; vertices: number; triangles: number }) {
  const currentTime = performance.now()
  const deltaTime = currentTime - fpsState.lastTime
  fpsState.lastTime = currentTime
  fpsState.frameTime = deltaTime
  fpsState.frameCount++

  // Update WebGL stats if provided
  if (webglStats) {
    fpsState.drawCalls = webglStats.drawCalls
    fpsState.vertices = webglStats.vertices
    fpsState.triangles = webglStats.triangles
  }

  // Update FPS every 100ms for smoother display
  if (currentTime - fpsState.lastFpsUpdate >= 100) {
    const fps = Math.round((fpsState.frameCount * 1000) / (currentTime - fpsState.lastFpsUpdate))
    fpsState.fps = fps

    // Update history
    fpsState.fpsHistory.push(fps)
    if (fpsState.fpsHistory.length > fpsState.maxHistoryLength) {
      fpsState.fpsHistory.shift()
    }

    // Calculate stats
    if (fps < fpsState.minFps) fpsState.minFps = fps
    if (fps > fpsState.maxFps) fpsState.maxFps = fps
    fpsState.avgFps = Math.round(fpsState.fpsHistory.reduce((a, b) => a + b, 0) / fpsState.fpsHistory.length)

    // Update display
    updateFpsDisplay()

    // Reset counters
    fpsState.frameCount = 0
    fpsState.lastFpsUpdate = currentTime
  }
}

// Update the display
function updateFpsDisplay() {
  const display = document.getElementById('fps-display')
  if (!display || !fpsState.showFps) return

  const fps = fpsState.fps

  // Update content based on mode
  if (fpsState.showAdvancedMetrics) {
    display.innerHTML = `
      <div class="fps-main">FPS: ${fps}</div>
      <div class="fps-detail">Min: ${fpsState.minFps} | Max: ${fpsState.maxFps} | Avg: ${fpsState.avgFps}</div>
      <div class="fps-detail">Frame: ${fpsState.frameTime.toFixed(1)}ms</div>
      <div class="fps-detail">Draw Calls: ${fpsState.drawCalls} | Triangles: ${fpsState.triangles}</div>
    `
    display.classList.add('advanced')
  } else {
    display.innerHTML = `<div class="fps-main">FPS: ${fps}</div>`
    display.classList.remove('advanced')
  }

  // Update color coding
  display.classList.remove('low-fps', 'critical-fps')
  if (fps < 30) {
    display.classList.add('critical-fps')
  } else if (fps < 60) {
    display.classList.add('low-fps')
  }
}

// Toggle FPS display
function toggleFpsDisplay() {
  fpsState.showFps = !fpsState.showFps
  saveFpsPreferences()

  const display = document.getElementById('fps-display')
  const button = document.getElementById('fps-toggle')

  if (display) {
    if (fpsState.showFps) {
      display.classList.remove('hidden')
    } else {
      display.classList.add('hidden')
    }
  }

  if (button) {
    button.style.opacity = fpsState.showFps ? '1' : '0.5'
  }

  requestFrame()
}

// Toggle advanced metrics
function toggleAdvancedMetrics() {
  fpsState.showAdvancedMetrics = !fpsState.showAdvancedMetrics
  saveFpsPreferences()
  updateFpsDisplay()
}

// Reset statistics
function resetFpsStats() {
  fpsState.minFps = Infinity
  fpsState.maxFps = 0
  fpsState.fpsHistory = []
  updateFpsDisplay()
}

// Inject FPS counter into DOM
function injectFpsCounter() {
  // Check if already injected
  if (document.getElementById('fps-display')) {
    return
  }

  // Create the FPS toggle button HTML
  const toggleButtonHTML = `
    <button id="fps-toggle" class="toggle-button" title="Toggle FPS display (F3 for advanced)">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <line x1="9" y1="9" x2="15" y2="9"></line>
        <line x1="9" y1="12" x2="15" y2="12"></line>
        <line x1="9" y1="15" x2="11" y2="15"></line>
      </svg>
    </button>
  `

  // Try to insert the button in the best location
  const glyphToggle = document.getElementById('glyph-toggle')

  // Insert after glyph toggle
  glyphToggle!.insertAdjacentHTML('afterend', toggleButtonHTML)

  // Create the FPS display HTML
  const fpsDisplayHTML = `
    <div id="fps-display" class="fps-display ${fpsState.showFps ? '' : 'hidden'}">
      <div class="fps-main">FPS: 0</div>
    </div>
  `

  // Add the FPS display to the body
  document.body.insertAdjacentHTML('beforeend', fpsDisplayHTML)

  // Add styles if not already present
  if (!document.getElementById('fps-counter-styles')) {
    const styleElement = document.createElement('style')
    styleElement.id = 'fps-counter-styles'
    styleElement.textContent = FPS_COUNTER_STYLES
    document.head.appendChild(styleElement)
  }

  // Set initial button state
  const button = document.getElementById('fps-toggle')
  if (button) {
    button.style.opacity = fpsState.showFps ? '1' : '0.5'
  }

  // Update display initially
  updateFpsDisplay()
}

// Initialize the FPS counter
export function initFpsCounter() {
  loadFpsPreferences()
  injectFpsCounter()

  // Attach event handlers
  onEvent('click', '#fps-toggle', () => {
    toggleFpsDisplay()
  })

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // F3 toggles advanced metrics
    if (e.key === 'F3') {
      e.preventDefault()
      if (fpsState.showFps) {
        toggleAdvancedMetrics()
      } else {
        // If FPS is hidden, show it with advanced metrics
        fpsState.showFps = true
        fpsState.showAdvancedMetrics = true
        saveFpsPreferences()
        const display = document.getElementById('fps-display')
        if (display) {
          display.classList.remove('hidden')
        }
        updateFpsDisplay()
      }
    }

    // Shift+F3 resets stats
    if (e.key === 'F3' && e.shiftKey) {
      e.preventDefault()
      resetFpsStats()
    }
  })

  // Context menu for FPS display
  onEvent('contextmenu', '#fps-display', (target: HTMLElement, e: Event) => {
    e.preventDefault()
    toggleAdvancedMetrics()
  })
}

// Get current WebGL stats from Context3d
export function getWebGLStats(ctx: any): { drawCalls: number; vertices: number; triangles: number } {
  let drawCalls = 0
  let vertices = 0
  let triangles = 0

  if (ctx && ctx.meshes) {
    for (const mesh of ctx.meshes.values()) {
      if (mesh.currentQuad > 0) {
        drawCalls++
        vertices += mesh.currentVertex
        triangles += mesh.currentQuad * 2
      }
    }
  }

  return { drawCalls, vertices, triangles }
}

// CSS styles
const FPS_COUNTER_STYLES = `
/* FPS Display */
.fps-display {
  position: fixed;
  top: 70px;
  right: 10px;
  background-color: rgba(0, 0, 0, 0.8);
  color: #00ff00;
  padding: 8px 12px;
  border-radius: 6px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  font-weight: bold;
  z-index: 1000;
  pointer-events: auto;
  user-select: none;
  cursor: pointer;
  border: 1px solid rgba(0, 255, 0, 0.2);
  backdrop-filter: blur(4px);
  transition: all 0.3s ease;
}

.fps-display:hover {
  background-color: rgba(0, 0, 0, 0.9);
  border-color: rgba(0, 255, 0, 0.4);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

.fps-display.hidden {
  display: none;
}

.fps-display.advanced {
  min-width: 220px;
}

.fps-main {
  font-size: 16px;
  margin-bottom: 4px;
}

.fps-detail {
  font-size: 11px;
  opacity: 0.8;
  margin: 2px 0;
}

/* Color coding */
.fps-display.low-fps {
  color: #ffff00;
  border-color: rgba(255, 255, 0, 0.2);
}

.fps-display.low-fps:hover {
  border-color: rgba(255, 255, 0, 0.4);
}

.fps-display.critical-fps {
  color: #ff3333;
  border-color: rgba(255, 51, 51, 0.2);
  animation: pulse 2s infinite;
}

.fps-display.critical-fps:hover {
  border-color: rgba(255, 51, 51, 0.4);
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* FPS Toggle Button */
#fps-toggle {
  transition: all 0.2s ease;
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 8px;
  border-radius: 4px;
}

#fps-toggle:hover {
  background-color: rgba(0, 255, 0, 0.1);
}

#fps-toggle svg {
  width: 20px;
  height: 20px;
  stroke: #ccc;
}

#fps-toggle:hover svg {
  stroke: #fff;
}

/* Responsive */
@media (max-width: 768px) {
  .fps-display {
    top: 65px;
    right: 5px;
    padding: 6px 10px;
    font-size: 12px;
  }

  .fps-main {
    font-size: 14px;
  }

  .fps-detail {
    font-size: 10px;
  }
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
  .fps-display {
    background-color: rgba(20, 20, 20, 0.9);
  }
}
`
