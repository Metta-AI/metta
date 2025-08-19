import { SkyJob } from '../types'

const API_BASE = 'http://localhost:8000'

export async function fetchSkyJobs(): Promise<SkyJob[]> {
  try {
    const response = await fetch(`${API_BASE}/api/sky-jobs`)
    if (!response.ok) {
      throw new Error('Failed to fetch sky jobs')
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Error fetching sky jobs:', error)
    // Return empty array if backend is not running or sky CLI not available
    return []
  }
}

export async function cancelSkyJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/sky-jobs/${jobId}/cancel`, {
    method: 'POST'
  })
  if (!response.ok) {
    throw new Error('Failed to cancel job')
  }
}

export async function launchWorker(numGpus: number, sweepName: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/sky-jobs/launch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ 
      num_gpus: numGpus, 
      sweep_name: sweepName 
    })
  })
  if (!response.ok) {
    throw new Error('Failed to launch worker')
  }
}