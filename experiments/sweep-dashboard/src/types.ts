export interface SweepRun {
  id: string
  name: string
  score: number
  cost: number
  runtime: number
  timestamp: string
  parameters: Record<string, any>
  status: 'running' | 'finished' | 'failed' | 'cancelled'
}

export interface ActiveRun {
  run_id: string
  run_name: string
  state: string
  created_at: string
  runtime_seconds: number
  timesteps: number
  total_timesteps: number
  progress: number
  score: number
  cost: number
  seconds_since_update?: number | null
}

export interface SweepData {
  runs: SweepRun[]
  activeRuns?: ActiveRun[]
  totalRuns: number
  bestScore: number
  totalCost: number
  avgRuntime: number
  parameters: string[]
}

export interface SkyJob {
  id: string
  name: string
  resources: string
  submitted: string
  totalDuration: string
  jobDuration: string
  status: 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'PENDING' | 'CANCELLED'
}

export interface SweepConfig {
  entity: string
  project: string
  sweepName: string
  maxObservations: number
  hourlyCost: number
}