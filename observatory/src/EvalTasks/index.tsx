import { FC, useCallback, useEffect, useState } from 'react'
import { EvalTask, Repo, TaskFilters } from '../repo'
import { TaskRow } from './TaskRow'

interface Props {
  repo: Repo
}

const pageSize = 50

export const EvalTasks: FC<Props> = ({ repo }) => {
  // State
  const [tasks, setTasks] = useState<EvalTask[]>([])
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [filters, setFilters] = useState<TaskFilters>({})

  // Form state
  const [command, setCommand] = useState('')
  const [gitHash, setGitHash] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load tasks
  const loadTasks = useCallback(async (page: number) => {
    try {
      const response = await repo.getEvalTasksPaginated(page, pageSize, filters)
      setTasks(response.tasks)
      setCurrentPage(response.page)
      setTotalPages(response.total_pages)
      setTotalCount(response.total_count)
    } catch (err: any) {
      console.error('Failed to load tasks:', err)
      setError(`Failed to load tasks: ${err.message}`)
    }
  }, [])

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

  // Create task
  const handleCreateTask = async () => {
    if (!command.trim()) {
      setError('Please enter a command')
      return
    }

    setLoading(true)
    setError(null)

    try {
      await repo.createEvalTask({
        command: command.trim(),
        git_hash: gitHash.trim() || null,
        attributes: {},
      })

      setCommand('')
      setGitHash('')
      await loadTasks(1)
    } catch (err: any) {
      setError(`Failed to create task: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

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

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '30px' }}>Evaluation Tasks</h1>

      {error && (
        <div
          style={{
            padding: '10px 15px',
            marginBottom: '20px',
            backgroundColor: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '4px',
            color: '#721c24',
          }}
        >
          {error}
        </div>
      )}

      {/* Create Task Section */}
      <div
        style={{
          backgroundColor: '#ffffff',
          padding: '24px',
          borderRadius: '12px',
          marginBottom: '30px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          border: '1px solid #e8e8e8',
        }}
      >
        <h3 className="mt-0 mb-5">Create New Task</h3>

        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <label className="block mb-1 text-sm font-medium">Command</label>
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="Enter command to execute"
              className="box-border"
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
              }}
            />
          </div>

          <div>
            <label className="block mb-1 text-sm font-medium">Git Hash (optional)</label>
            <input
              type="text"
              value={gitHash}
              onChange={(e) => setGitHash(e.target.value)}
              placeholder="Git commit hash"
              className="box-border"
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                fontSize: '14px',
              }}
            />
          </div>

          <button
            onClick={handleCreateTask}
            disabled={loading || !command.trim()}
            style={{
              padding: '10px 24px',
              backgroundColor: loading || !command.trim() ? '#9ca3af' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !command.trim() ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 500,
            }}
          >
            {loading ? 'Creating...' : 'Create Task'}
          </button>
        </div>
      </div>

      {/* Tasks Table */}
      <div>
        <h2 className="mb-5">All Tasks ({totalCount})</h2>
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
                  {renderFilterInput(filters.created_at || '', (value) =>
                    setFilters({ ...filters, created_at: value })
                  )}
                </th>
                <th style={{ padding: '12px', borderBottom: '2px solid #dee2e6', width: '8%' }}>Logs</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((task) => (
                <TaskRow key={task.id} task={task} repo={repo} />
              ))}
            </tbody>
          </table>
          {tasks.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No tasks found</div>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex gap-2 justify-center py-5">
            <button
              onClick={() => loadTasks(currentPage - 1)}
              disabled={currentPage === 1}
              className="px-3 py-2 border border-gray-300 rounded-md cursor-pointer disabled:cursor-not-allowed disabled:bg-gray-100"
            >
              Previous
            </button>
            <span className="px-3 py-2">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => loadTasks(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="px-3 py-2 border border-gray-300 rounded-md cursor-pointer disabled:cursor-not-allowed disabled:bg-gray-100"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
