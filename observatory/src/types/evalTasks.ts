export type SortField =
  | 'policy_name'
  | 'sim_suite'
  | 'status'
  | 'assignee'
  | 'user_id'
  | 'retries'
  | 'created_at'
  | 'assigned_at'
  | 'updated_at'

export type SortDirection = 'asc' | 'desc'

export interface EvalTasksProps {
  repo: any // Repo type from repo.ts
}

export interface SortHeaderProps {
  field: SortField
  label: string
  isActive: boolean
  width?: string
  onSort: (field: SortField, isActive: boolean) => void
  activeSortField: SortField
  activeSortDirection: SortDirection
  completedSortField: SortField
  completedSortDirection: SortDirection
}

export interface TaskRowProps {
  task: any // EvalTask type from repo.ts
  isActive: boolean
  isExpanded: boolean
  onToggleExpansion: (taskId: string) => void
  getDisplayStatus: (task: any) => string
  getWorkingDuration: (task: any) => string | null
  getStatusColor: (status: string, isInProgress: boolean) => string
  getWorkerColor: (workerName: string | null) => string
  truncateWorkerName: (workerName: string | null) => string
  getGithubUrl: (gitHash: string) => string
}

export interface AttributesRendererProps {
  attributes: Record<string, any>
}
