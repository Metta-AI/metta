import { state, HEATMAP_MIN_OPACITY, HEATMAP_MAX_OPACITY } from './common.js'

/**
 * Tracks agent presence on tiles over time.
 */
export class Heatmap {
  private data: number[][][] = []
  private width: number = 0
  private height: number = 0
  private maxSteps: number = 0

  initialize(): void {
    this.width = state.replay.mapSize[0]
    this.height = state.replay.mapSize[1]
    this.maxSteps = state.replay.maxSteps
    console.info('Heatmap initialized:', this.width, 'x', this.height, 'x', this.maxSteps)

    this.data = []
    for (let step = 0; step < this.maxSteps; step++) {
      const stepData: number[][] = []
      for (let x = 0; x < this.width; x++) {
        stepData[x] = new Array(this.height).fill(0)
      }
      this.data[step] = stepData
    }

    // Initialize the heatmap for every step in the current replay.
    for (let step = 0; step < this.maxSteps; step++) {
      // Start with previous step's cumulative values.
      if (step > 0) {
        for (let x = 0; x < this.width; x++) {
          for (let y = 0; y < this.height; y++) {
            this.data[step][x][y] = this.data[step - 1][x][y]
          }
        }
      }

      // Add agent positions for this step.
      for (const agent of state.replay.agents) {
        const location = agent.location.get(step)
        const x = location[0]
        const y = location[1]

        this.assertValidPosition(step, x, y)
        this.data[step][x][y]++
      }
    }
  }

  /**
   * In Play mode, we need to update the heatmap for every new step.
   */
  update(step: number): void {
    // expand the data structure for the new step.
    if (step >= this.maxSteps) {
      const oldMaxSteps = this.maxSteps
      this.maxSteps = step + 1

      for (let i = oldMaxSteps; i < this.maxSteps; i++) {
        const stepData: number[][] = []
        for (let x = 0; x < this.width; x++) {
          stepData[x] = new Array(this.height).fill(0)
        }

        // Start with previous step's cumulative values.
        if (i > 0) {
          for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
              stepData[x][y] = this.data[i - 1][x][y]
            }
          }
        }

        this.data[i] = stepData
      }
    }

    // Update the cumulative heatmap for every agent at this step.
    for (const agent of state.replay.agents) {
      const location = agent.location.get(step)
      const x = location[0]
      const y = location[1]

      this.assertValidPosition(step, x, y)
      this.data[step][x][y]++
    }
  }

  assertValidPosition(step: number, x: number, y: number): void {
    if (step < 0 || step >= this.maxSteps || x < 0 || x >= this.width || y < 0 || y >= this.height) {
      throw new Error(`Invalid heatmap position: ${step}, ${x}, ${y}`)
    }
  }

  getHeat(step: number, x: number, y: number): number {
    this.assertValidPosition(step, x, y)
    return this.data[step][x][y]
  }
}

/** Maps a normalized heat value (0-1) to a color using a blue-yellow-red thermal gradient. */
export function getHeatmapColor(normalizedHeat: number): [number, number, number, number] {
  // Use square root scaling for better visual distribution.
  const scaledHeat = Math.sqrt(normalizedHeat)
  const opacity = HEATMAP_MIN_OPACITY * 0.5 + scaledHeat * (HEATMAP_MAX_OPACITY * 0.6)
  let r: number, g: number, b: number

  if (scaledHeat < 0.5) {
    // Blue to yellow transition: blue → cyan → green → yellow.
    const t = scaledHeat * 2 // Normalize to 0-1 range for this half.
    r = Math.max((t - 0.5) * 1.6, 0) // Only add red in second half (0.5-1.0) to avoid purple.
    g = t * 0.8 // Add green from 0 to 0.8 for cyan/green.
    b = (1 - t) * 0.7 // Fade out blue linearly from 0.7 to 0.
  } else {
    // Yellow to red transition preserves red and reduces green.
    const t = (scaledHeat - 0.5) * 2 // Normalize to 0-1 range for this half.
    r = 0.8 + t * 0.2 // Increase red slightly from 0.8 to 1.0.
    g = 0.8 * (1 - t) // Fade out green from 0.8 to 0 for pure red.
    b = 0 // No blue component in yellow-red range.
  }

  return [r, g, b, opacity]
}

/** Gets the maximum heat value for a given step. */
export function getMaxHeat(step: number): number {
  let maxHeat = 0
  for (let x = 0; x < state.replay.mapSize[0]; x++) {
    for (let y = 0; y < state.replay.mapSize[1]; y++) {
      const heat = state.heatmap.getHeat(step, x, y)
      maxHeat = Math.max(maxHeat, heat)
    }
  }
  return maxHeat
}

/** Renders heatmap tiles using a provided drawing function. */
export function renderHeatmapTiles(
  step: number,
  drawTile: (x: number, y: number, color: [number, number, number, number]) => void
): void {
  const maxHeat = getMaxHeat(step)
  if (maxHeat === 0) return

  for (let x = 0; x < state.replay.mapSize[0]; x++) {
    for (let y = 0; y < state.replay.mapSize[1]; y++) {
      const heat = state.heatmap.getHeat(step, x, y)
      if (heat > 0) {
        const normalizedHeat = heat / maxHeat
        const color = getHeatmapColor(normalizedHeat)
        drawTile(x, y, color)
      }
    }
  }
}
