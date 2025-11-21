import { FC, PropsWithChildren, Ref, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'

import { Button } from '../components/Button'
import { Input } from '../components/Input'
import { PaginatedEvalTasksResponse, Repo, TaskFilters } from '../repo'
import { TaskRow } from './TaskRow'

const pageSize = 50

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

const TH: FC<
  PropsWithChildren<{
    style?: React.CSSProperties
  }>
> = ({ children, style }) => {
  return (
    <th
      className="px-3 pt-2 pb-0.5 text-left text-xs text-gray-800 font-semibold tracking-wide uppercase"
      style={style}
    >
      {children}
    </th>
  )
}

const THFilter: FC<PropsWithChildren> = ({ children }) => {
  return <th className="px-1 pb-2">{children}</th>
}

export const TasksTable: FC<{
  repo: Repo
  setError: (error: string) => void
  ref?: Ref<TasksTableHandle>
}> = ({ repo, setError, ref }) => {
  const [tasksResponse, setTasksResponse] = useState<PaginatedEvalTasksResponse | undefined>()
  const currentPage = tasksResponse?.page || 1
  const [filters, setFilters] = useState<TaskFilters>({})
  const isInitialMount = useRef(true)

  // Load tasks
  const loadTasks = useCallback(
    async (page: number) => {
      try {
        const response = await repo.getEvalTasksPaginated(page, pageSize, filters)
        setTasksResponse(response)
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
    return <div>Loading tasks...</div>
  }

  return (
    <div>
      <h2 className="mb-5">All Tasks ({tasksResponse.total_count})</h2>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse table-fixed">
          <thead className="border-b border-b-gray-300 bg-gray-100">
            <tr>
              <TH>ID</TH>
              <TH style={{ width: '30%' }}>Command</TH>
              <TH>Status</TH>
              <TH>User</TH>
              <TH>Assignee</TH>
              <TH>Attempts</TH>
              <TH>Created</TH>
              <TH>Logs</TH>
            </tr>
            <tr>
              <THFilter />
              <THFilter>
                <FilterInput
                  value={filters.command || ''}
                  onChange={(value) => setFilters({ ...filters, command: value })}
                />
              </THFilter>
              <THFilter>
                <StatusDropdown
                  value={filters.status || ''}
                  onChange={(value) => setFilters({ ...filters, status: value })}
                />
              </THFilter>
              <THFilter>
                <FilterInput
                  value={filters.user_id || ''}
                  onChange={(value) => setFilters({ ...filters, user_id: value })}
                />
              </THFilter>
              <THFilter>
                <FilterInput
                  value={filters.assignee || ''}
                  onChange={(value) => setFilters({ ...filters, assignee: value })}
                />
              </THFilter>
              <THFilter />
              <THFilter>
                <FilterInput
                  value={filters.created_at || ''}
                  onChange={(value) => setFilters({ ...filters, created_at: value })}
                />
              </THFilter>
              <th />
            </tr>
          </thead>
          <tbody>
            {tasksResponse.tasks.map((task) => (
              <TaskRow key={task.id} task={task} repo={repo} />
            ))}
          </tbody>
        </table>
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
