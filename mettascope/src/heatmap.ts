import { state } from './common.js'
import { getAttr } from './replay.js'

/**
 * Tracks agent presence on tiles over time.
 */
export class Heatmap {
  private data: number[][][] = []
  private width: number = 0
  private height: number = 0
  private maxSteps: number = 0

  initialize(): void {
    this.width = state.replay.map_size[0]
    this.height = state.replay.map_size[1]
    this.maxSteps = state.replay.max_steps
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
      for (const gridObject of state.replay.grid_objects) {
        if (gridObject.agent_id === undefined) continue

        const x = getAttr(gridObject, 'c', step)
        const y = getAttr(gridObject, 'r', step)

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
    for (const gridObject of state.replay.grid_objects) {
      if (gridObject.agent_id === undefined) continue

      const x = getAttr(gridObject, 'c', step)
      const y = getAttr(gridObject, 'r', step)

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
