import { EvalTask } from '../repo'
import { SortField, SortDirection } from '../types/evalTasks'
import { METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO } from '../constants'

export const getStatusColor = (status: string, isInProgress: boolean): string => {
  if (isInProgress) return '#17a2b8' // info blue for in progress
  switch (status) {
    case 'done':
      return '#28a745'
    case 'error':
      return '#dc3545'
    case 'unprocessed':
      return '#6c757d'
    case 'canceled':
      return '#ffc107'
    default:
      return '#6c757d'
  }
}

export const getDisplayStatus = (task: EvalTask): string => {
  // Only show "in progress" for unprocessed tasks with recent assignment
  if (task.status === 'unprocessed' && task.assignee && task.assigned_at) {
    const assignedTime = new Date(task.assigned_at + 'Z').getTime()
    const now = new Date().getTime()
    const twentyMinutesAgo = now - 20 * 60 * 1000
    if (assignedTime > twentyMinutesAgo) {
      return 'in progress'
    }
  }
  return task.status
}

export const getWorkingDuration = (task: EvalTask): string | null => {
  // Only show duration for tasks that display as "in progress"
  if (getDisplayStatus(task) !== 'in progress') return null

  if (!task.assigned_at) return null

  const start = new Date(task.assigned_at + 'Z') // Add Z to indicate UTC
  const now = new Date()
  const diff = now.getTime() - start.getTime()

  // If negative (assigned in future?), show 0
  if (diff < 0) return '00:00'

  const totalSeconds = Math.floor(diff / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  // Format as MM:SS
  const formattedMinutes = minutes.toString().padStart(2, '0')
  const formattedSeconds = seconds.toString().padStart(2, '0')

  return `${formattedMinutes}:${formattedSeconds}`
}

export const getGithubUrl = (gitHash: string): string => {
  // Assuming metta repo, adjust if needed
  return `https://github.com/${METTA_GITHUB_ORGANIZATION}/${METTA_GITHUB_REPO}/commit/${gitHash}`
}

export const truncateWorkerName = (workerName: string | null): string => {
  if (!workerName) return ''
  const parts = workerName.split('-')
  if (parts.length >= 3) {
    // Get the last part (suffix) and abbreviate the middle parts
    const suffix = parts[parts.length - 1]
    return suffix
  }
  return workerName
}

export const getWorkerColor = (workerName: string | null): string => {
  if (!workerName) return 'transparent'

  // Simple hash function to generate consistent colors
  let hash = 0
  for (let i = 0; i < workerName.length; i++) {
    hash = workerName.charCodeAt(i) + ((hash << 5) - hash)
  }

  // Generate a pastel color
  const hue = hash % 360
  return `hsl(${hue}, 70%, 85%)`
}

export const sortTasks = (tasksToSort: EvalTask[], field: SortField, direction: SortDirection): EvalTask[] => {
  return [...tasksToSort].sort((a, b) => {
    let aVal: any = a[field as keyof EvalTask]
    let bVal: any = b[field as keyof EvalTask]

    // Special handling for status to show in-progress correctly
    if (field === 'status') {
      aVal = getDisplayStatus(a)
      bVal = getDisplayStatus(b)
    }

    // Handle git hash sorting
    if (field === 'policy_name') {
      // Secondary sort by git hash
      if (aVal === bVal) {
        aVal = a.attributes?.git_hash || ''
        bVal = b.attributes?.git_hash || ''
      }
    }

    if (aVal === null || aVal === undefined) aVal = ''
    if (bVal === null || bVal === undefined) bVal = ''

    if (aVal < bVal) return direction === 'asc' ? -1 : 1
    if (aVal > bVal) return direction === 'asc' ? 1 : -1
    return 0
  })
}

export const getRecentPolicies = (tasks: EvalTask[]): string[] => {
  const policySet = new Set<string>()
  tasks.forEach((task) => {
    if (task.policy_name) policySet.add(task.policy_name)
    policySet.add(task.policy_id)
  })
  return Array.from(policySet).sort()
}

export const getRecentSimSuites = (tasks: EvalTask[]): string[] => {
  const simSuiteSet = new Set<string>()
  tasks.forEach((task) => {
    if (task.sim_suite) simSuiteSet.add(task.sim_suite)
  })
  return Array.from(simSuiteSet).sort()
}
