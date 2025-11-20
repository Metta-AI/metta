import { FC, Ref, useCallback, useEffect, useImperativeHandle, useState } from 'react'
import { PaginatedEvalTasksResponse, Repo, TaskFilters } from '../repo'
import { TaskRow } from './TaskRow'

const pageSize = 50

export type TasksTableHandle = {
  loadTasks: (page: number) => void
}

export const TasksTable: FC<{
  repo: Repo
  setError: (error: string) => void
  ref?: Ref<TasksTableHandle>
}> = ({ repo, setError, ref }) => {
  const [tasksResponse, setTasksResponse] = useState<PaginatedEvalTasksResponse | undefined>()
  const currentPage = tasksResponse?.page || 1
  const [filters, setFilters] = useState<TaskFilters>({})

  // Load tasks
  const loadTasks = useCallback(async (page: number) => {
    try {
      const response = await repo.getEvalTasksPaginated(page, pageSize, filters)
      setTasksResponse(response)
    } catch (err: any) {
      console.error('Failed to load tasks:', err)
      setError(`Failed to load tasks: ${err.message}`)
    }
  }, [])

  useImperativeHandle(ref, () => ({
    loadTasks,
  }))

  // Initial load
  useEffect(() => {
    loadTasks(1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Reload when filters change
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      loadTasks(1)
    }, 300)
    return () => clearTimeout(timeoutId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadTasks(currentPage)
    }, 5000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, filters])

  // Render helpers
  const renderFilterInput = (value: string, onChange: (value: string) => void, placeholder: string = 'Filter...') => {
    return (
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          padding: '4px 8px',
          fontSize: '12px',
          border: '1px solid #d1d5db',
          borderRadius: '4px',
          marginTop: '4px',
        }}
      />
    )
  }

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
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '5%' }}>ID</th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '30%' }}>
                Command
                {renderFilterInput(filters.command || '', (value) => setFilters({ ...filters, command: value }))}
              </th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '10%' }}>
                Status
                {renderStatusDropdown(filters.status || '', (value) => setFilters({ ...filters, status: value }))}
              </th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                User
                {renderFilterInput(filters.user_id || '', (value) => setFilters({ ...filters, user_id: value }))}
              </th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                Assignee
                {renderFilterInput(filters.assignee || '', (value) => setFilters({ ...filters, assignee: value }))}
              </th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '8%' }}>Attempts</th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '15%' }}>
                Created
                {renderFilterInput(filters.created_at || '', (value) => setFilters({ ...filters, created_at: value }))}
              </th>
              <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '8%' }}>Logs</th>
            </tr>
          </thead>
          <tbody>
            {tasksResponse.tasks.map((task) => (
              <TaskRow key={task.id} task={task} repo={repo} />
            ))}
          </tbody>
        </table>
        {tasksResponse.tasks.length === 0 && (
          <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No tasks found</div>
        )}
      </div>

      {/* Pagination */}
      {tasksResponse.total_pages > 1 && (
        <div className="flex gap-2 justify-center py-5">
          <button
            onClick={() => loadTasks(currentPage - 1)}
            disabled={currentPage === 1}
            className="px-3 py-2 border border-gray-300 rounded-md cursor-pointer disabled:cursor-not-allowed disabled:bg-gray-100"
          >
            Previous
          </button>
          <span className="px-3 py-2">
            Page {currentPage} of {tasksResponse.total_pages}
          </span>
          <button
            onClick={() => loadTasks(currentPage + 1)}
            disabled={currentPage === tasksResponse.total_pages}
            className="px-3 py-2 border border-gray-300 rounded-md cursor-pointer disabled:cursor-not-allowed disabled:bg-gray-100"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
