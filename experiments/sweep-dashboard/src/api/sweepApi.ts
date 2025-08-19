import { SweepData, SweepRun } from '../types'

const API_BASE = 'http://localhost:8000'

export async function fetchAvailableSweeps(): Promise<string[]> {
  try {
    const response = await fetch(`${API_BASE}/api/sweeps`)
    if (!response.ok) {
      throw new Error('Failed to fetch sweeps')
    }
    const data = await response.json()
    return data.sweeps
  } catch (error) {
    console.error('Error fetching sweeps:', error)
    // Return empty array if backend is not running
    return []
  }
}

export async function fetchSweepData(sweepName: string): Promise<SweepData> {
  try {
    const response = await fetch(`${API_BASE}/api/sweeps/${sweepName}`)
    if (!response.ok) {
      throw new Error('Failed to fetch sweep data')
    }
    const data = await response.json()
    
    // Transform the data to match our frontend types
    return {
      runs: data.runs.map((run: any) => ({
        id: run.id,
        name: run.name,
        score: run.score,
        cost: run.cost,
        runtime: run.runtime,
        timestamp: run.timestamp,
        parameters: run.parameters || {},
        status: run.status
      })),
      activeRuns: data.activeRuns || [],
      totalRuns: data.totalRuns,
      bestScore: data.bestScore,
      totalCost: data.totalCost,
      avgRuntime: data.avgRuntime,
      parameters: data.parameters
    }
  } catch (error) {
    console.error('Error fetching sweep data:', error)
    throw error
  }
}

// For production, replace with actual WandB API integration
export async function fetchFromWandB(entity: string, project: string, sweepName: string) {
  const response = await fetch(`/api/wandb/sweeps/${entity}/${project}/${sweepName}`)
  if (!response.ok) {
    throw new Error('Failed to fetch sweep data from WandB')
  }
  return response.json()
}