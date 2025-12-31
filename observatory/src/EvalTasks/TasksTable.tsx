import { FC, Ref, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'

import { Button } from '../components/Button'
import { Input } from '../components/Input'
import { Spinner } from '../components/Spinner'
import { Table, TH, TR } from '../components/Table'
import { PaginatedEvalTasksResponse, PublicPolicyVersionRow, Repo, TaskFilters } from '../repo'
import { TaskRow } from './TaskRow'

const pageSize = 50

const UUID_REGEX = /[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/

export function parsePolicyVersionId(command: string): string | null {
  const match = command.match(/policy_version_id=([0-9a-fA-F-]{36})/)
  if (match && UUID_REGEX.test(match[1])) {
    return match[1]
  }
  return null
}

export type TasksTableHandle = {
  loadTasks: (page: number) => void
}

const FilterInput: FC<{
  value: string
  onChange: (value: string) => void
}> = ({ value, onChange }) => {
  return <Input value={value} onChange={onChange} placeholder="Filter..." size="sm" />
}

const StatusDropdown: FC<{ value: string; onChange: (value: string) => void }> = ({ value, onChange }) => {
  return (
    <select
      value={value || ''}
      onChange={(e) => onChange(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      className="w-full rounded border h-6 border-gray-300 bg-white text-gray-800 text-xs py-1 pl-1"
    >
      <option value="">All</option>
      <option value="unprocessed">Unprocessed</option>
      <option value="running">Running</option>
      <option value="done">Done</option>
      <option value="error">Error</option>
      <option value="system_error">System Error</option>
      <option value="canceled">Canceled</option>
    </select>
  )
}

export const TasksTable: FC<{
  repo: Repo
  setError: (error: string) => void
  ref?: Ref<TasksTableHandle>
  initialFilters?: TaskFilters
  hideFilters?: boolean
}> = ({ repo, setError, ref, initialFilters, hideFilters }) => {
  const [tasksResponse, setTasksResponse] = useState<PaginatedEvalTasksResponse | undefined>()
  const currentPage = tasksResponse?.page || 1
  const [filters, setFilters] = useState<TaskFilters>(initialFilters || {})
  const isInitialMount = useRef(true)
  const [policyInfoMap, setPolicyInfoMap] = useState<Record<string, PublicPolicyVersionRow>>({})
  const attemptedPolicyIds = useRef<Set<string>>(new Set())

  // Load tasks
  const loadTasks = useCallback(
    async (page: number) => {
      try {
        const response = await repo.getEvalTasksPaginated(page, pageSize, filters)
        setTasksResponse(response)

        // Extract unique policy_version_ids from commands that we haven't attempted yet
        const policyVersionIds: string[] = []
        for (const task of response.tasks) {
          const pvId = parsePolicyVersionId(task.command)
          if (pvId && !attemptedPolicyIds.current.has(pvId)) {
            policyVersionIds.push(pvId)
          }
        }

        // Batch fetch policy info for new IDs
        if (policyVersionIds.length > 0) {
          try {
            const policyVersions = await repo.getPolicyVersionsBatch(policyVersionIds)
            // Mark as attempted only after successful fetch
            for (const pvId of policyVersionIds) {
              attemptedPolicyIds.current.add(pvId)
            }
            const newPolicyInfo: Record<string, PublicPolicyVersionRow> = {}
            for (const pv of policyVersions) {
              newPolicyInfo[pv.id] = pv
            }
            if (Object.keys(newPolicyInfo).length > 0) {
              setPolicyInfoMap((prev) => ({ ...prev, ...newPolicyInfo }))
            }
          } catch (err) {
            console.error('Failed to fetch policy versions:', err)
          }
        }
      } catch (err: any) {
        console.error('Failed to load tasks:', err)
        setError(`Failed to load tasks: ${err.message}`)
      }
    },
    [repo, setError, filters]
  )

  useImperativeHandle(ref, () => ({
    loadTasks,
  }))

  // Initial load and filter changes (with 300ms debounce for filter changes)
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false
      loadTasks(1)
      return
    }

    const timeoutId = setTimeout(() => {
      loadTasks(1)
    }, 300)
    return () => clearTimeout(timeoutId)
  }, [loadTasks])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadTasks(currentPage)
    }, 5000)
    return () => clearInterval(interval)
  }, [loadTasks, currentPage])

  if (!tasksResponse) {
    return <Spinner size="lg" />
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <Table>
          <Table.Header>
            <TR>
              <TH>Policy</TH>
              <TH>
                <div className="flex flex-col gap-1">
                  Recipe
                  {!hideFilters && (
                    <FilterInput
                      value={filters.command || ''}
                      onChange={(value) => setFilters({ ...filters, command: value })}
                    />
                  )}
                </div>
              </TH>
              <TH>
                <div className="flex flex-col gap-1">
                  Status
                  {!hideFilters && (
                    <StatusDropdown
                      value={filters.status || ''}
                      onChange={(value) => setFilters({ ...filters, status: value })}
                    />
                  )}
                </div>
              </TH>
              <TH>Created</TH>
              <TH>Duration</TH>
              <TH>Logs</TH>
            </TR>
          </Table.Header>
          <Table.Body>
            {tasksResponse.tasks.map((task) => (
              <TaskRow
                key={task.id}
                task={task}
                repo={repo}
                policyInfoMap={policyInfoMap}
                attemptedPolicyIds={attemptedPolicyIds.current}
              />
            ))}
          </Table.Body>
        </Table>
        {tasksResponse.tasks.length === 0 && <div className="p-5 text-center text-gray-500">No tasks found</div>}
      </div>

      {/* Pagination */}
      {tasksResponse.total_pages > 1 && (
        <div className="flex gap-2 justify-center py-5">
          <Button onClick={() => loadTasks(currentPage - 1)} disabled={currentPage === 1}>
            Previous
          </Button>
          <span className="px-3 py-2 text-sm">
            Page {currentPage} of {tasksResponse.total_pages}
          </span>
          <Button onClick={() => loadTasks(currentPage + 1)} disabled={currentPage === tasksResponse.total_pages}>
            Next
          </Button>
        </div>
      )}
    </div>
  )
}
