import { getAttr } from './replay.js'

/**
 * Tracks agent presence on tiles over time using a 3D array structure: [timeStep][x][y].
 */
export class Heatmap {
  private data: number[][][]
  private width: number
  private height: number
  private maxSteps: number
  private initialized: boolean = false

  constructor(width: number, height: number, maxSteps: number) {
    this.width = width
    this.height = height
    this.maxSteps = maxSteps
    this.data = []
    this.initialize()
  }

  private initialize(): void {
    this.data = []
    for (let step = 0; step < this.maxSteps; step++) {
      const stepData: number[][] = []
      for (let x = 0; x < this.width; x++) {
        stepData[x] = new Array(this.height).fill(0)
      }
      this.data[step] = stepData
    }
    this.initialized = true
  }

  markTile(step: number, x: number, y: number): void {
    this.assertValidPosition(step, x, y)
    this.data[step][x][y] = 1
  }

  wasOccupied(step: number, x: number, y: number): boolean {
    this.assertValidPosition(step, x, y)
    return this.data[step][x][y] === 1
  }

  getStepData(step: number): number[][] | null {
    if (step < 0 || step >= this.maxSteps) {
      return null
    }
    return this.data[step]
  }

  getTileHistory(x: number, y: number): number[] {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return []
    }

    const history: number[] = []
    for (let step = 0; step < this.maxSteps; step++) {
      history.push(this.data[step][x][y])
    }
    return history
  }

  /**
   * Updates heatmap by marking tiles where agents are present for the current step.
   */
  updateFromGameState(step: number, gridObjects: any[]): void {
    if (!this.initialized || step < 0 || step >= this.maxSteps) {
      return
    }

    for (const gridObject of gridObjects) {
      if (gridObject.agent_id === undefined) continue

      const x = getAttr(gridObject, 'c')
      const y = getAttr(gridObject, 'r')

      if (x !== null && y !== null) {
        this.markTile(step, x, y)
      }
    }
  }

  private assertValidPosition(step: number, x: number, y: number): void {
    if (step < 0 || step >= this.maxSteps) {
      throw new Error(`Invalid step: ${step}, must be between 0 and ${this.maxSteps - 1}`)
    }
    if (x < 0 || x >= this.width) {
      throw new Error(`Invalid x coordinate: ${x}, must be between 0 and ${this.width - 1}`)
    }
    if (y < 0 || y >= this.height) {
      throw new Error(`Invalid y coordinate: ${y}, must be between 0 and ${this.height - 1}`)
    }
  }

  clear(): void {
    this.initialize()
  }

  getDimensions() {
    return {
      width: this.width,
      height: this.height,
      maxSteps: this.maxSteps
    }
  }
}
