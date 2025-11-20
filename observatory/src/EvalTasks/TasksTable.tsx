import { FC, Ref, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'

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

const TH: FC<{
  children: React.ReactNode
  style?: React.CSSProperties
}> = ({ children, style }) => {
  return (
    <th className="py-3 px-2 border-b border-b-gray-400" style={style}>
      {children}
    </th>
  )
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

  const renderStatusDropdown = (value: string, onChange: (value: string) => void) => {
    return (
      <select
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          padding: '4px 8px',
          fontSize: '12px',
          border: '1px solid #d1d5db',
          borderRadius: '4px',
          marginTop: '4px',
          backgroundColor: '#fff',
          cursor: 'pointer',
        }}
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

  if (!tasksResponse) {
    return <div>Loading tasks...</div>
  }

  return (
    <div>
      <h2 className="mb-5">All Tasks ({tasksResponse.total_count})</h2>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse table-fixed">
          <thead>
            <tr className="bg-gray-100 text-left">
              <TH>ID</TH>
              <TH style={{ width: '30%' }}>
                Command
                <FilterInput
                  value={filters.command || ''}
                  onChange={(value) => setFilters({ ...filters, command: value })}
                />
              </TH>
              <TH>
                Status
                {renderStatusDropdown(filters.status || '', (value) => setFilters({ ...filters, status: value }))}
              </TH>
              <TH>
                User
                <FilterInput
                  value={filters.user_id || ''}
                  onChange={(value) => setFilters({ ...filters, user_id: value })}
                />
              </TH>
              <TH>
                Assignee
                <FilterInput
                  value={filters.assignee || ''}
                  onChange={(value) => setFilters({ ...filters, assignee: value })}
                />
              </TH>
              <TH>Attempts</TH>
              <TH>
                Created
                <FilterInput
                  value={filters.created_at || ''}
                  onChange={(value) => setFilters({ ...filters, created_at: value })}
                />
              </TH>
              <TH>Logs</TH>
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
          <span className="px-3 py-2">
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
