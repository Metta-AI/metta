import React, { useState, useEffect } from 'react'
import { Repo, EvalTask, TaskAttempt, TaskFilters } from './repo'

interface Props {
  repo: Repo
}

export function EvalTasks({ repo }: Props) {
  // State
  const [tasks, setTasks] = useState<EvalTask[]>([])
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [filters, setFilters] = useState<TaskFilters>({})

  // UI state
  const [expandedTasks, setExpandedTasks] = useState<Set<number>>(new Set())
  const [taskAttempts, setTaskAttempts] = useState<Map<number, TaskAttempt[]>>(new Map())
  const [loadingAttempts, setLoadingAttempts] = useState<Set<number>>(new Set())

  // Form state
  const [command, setCommand] = useState('')
  const [gitHash, setGitHash] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const pageSize = 50

  // Load tasks
  const loadTasks = async (page: number) => {
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
  }

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

  // Toggle task expansion
  const toggleTaskExpansion = async (taskId: number) => {
    const newExpanded = new Set(expandedTasks)

    if (newExpanded.has(taskId)) {
      newExpanded.delete(taskId)
    } else {
      newExpanded.add(taskId)
      // Load attempts if not already loaded
      if (!taskAttempts.has(taskId) && !loadingAttempts.has(taskId)) {
        setLoadingAttempts(new Set(loadingAttempts).add(taskId))
        try {
          const response = await repo.getTaskAttempts(taskId)
          setTaskAttempts(new Map(taskAttempts).set(taskId, response.attempts))
        } catch (err) {
          console.error('Failed to load attempts:', err)
        } finally {
          setLoadingAttempts((prev) => {
            const next = new Set(prev)
            next.delete(taskId)
            return next
          })
        }
      }
    }

    setExpandedTasks(newExpanded)
  }

  // Render helpers
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'done':
        return '#28a745'
      case 'error':
      case 'system_error':
        return '#dc3545'
      case 'unprocessed':
        return '#6c757d'
      case 'running':
        return '#17a2b8'
      case 'canceled':
        return '#ffc107'
      default:
        return '#6c757d'
    }
  }

  const truncateCommand = (cmd: string, maxLength: number = 60) => {
    return cmd.length > maxLength ? cmd.substring(0, maxLength) + '...' : cmd
  }

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
        <h3 style={{ marginTop: 0, marginBottom: '20px' }}>Create New Task</h3>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end' }}>
          <div style={{ flex: '1' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', fontWeight: 500 }}>Command</label>
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="Enter command to execute"
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                fontSize: '14px',
              }}
            />
          </div>

          <div style={{ flex: '0 0 250px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', fontWeight: 500 }}>
              Git Hash (optional)
            </label>
            <input
              type="text"
              value={gitHash}
              onChange={(e) => setGitHash(e.target.value)}
              placeholder="Git commit hash"
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
        <h2 style={{ marginBottom: '20px' }}>All Tasks ({totalCount})</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa' }}>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '5%' }}>
                  ID
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '30%' }}>
                  Command
                  {renderFilterInput(filters.command || '', (value) => setFilters({ ...filters, command: value }))}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '10%' }}>
                  Status
                  {renderStatusDropdown(filters.status || '', (value) => setFilters({ ...filters, status: value }))}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                  User
                  {renderFilterInput(filters.user_id || '', (value) => setFilters({ ...filters, user_id: value }))}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                  Assignee
                  {renderFilterInput(filters.assignee || '', (value) => setFilters({ ...filters, assignee: value }))}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '8%' }}>
                  Attempts
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '15%' }}>
                  Created
                  {renderFilterInput(filters.created_at || '', (value) =>
                    setFilters({ ...filters, created_at: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '8%' }}>
                  Logs
                </th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((task) => {
                const isExpanded = expandedTasks.has(task.id)
                const attempts = taskAttempts.get(task.id) || []
                const hasMultipleAttempts = (task.attempt_number || 0) > 0

                return (
                  <React.Fragment key={task.id}>
                    <tr
                      style={{
                        borderBottom: '1px solid #dee2e6',
                        cursor: hasMultipleAttempts ? 'pointer' : 'default',
                      }}
                      onClick={() => hasMultipleAttempts && toggleTaskExpansion(task.id)}
                    >
                      <td style={{ padding: '12px' }}>
                        {hasMultipleAttempts && (
                          <span style={{ marginRight: '8px', fontSize: '12px', color: '#6c757d' }}>
                            {isExpanded ? '▼' : '▶'}
                          </span>
                        )}
                        {task.id}
                      </td>
                      <td style={{ padding: '12px' }} title={task.command}>
                        {truncateCommand(task.command)}
                      </td>
                      <td style={{ padding: '12px' }}>
                        <span
                          style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            backgroundColor: getStatusColor(task.status),
                            color: 'white',
                            fontSize: '12px',
                          }}
                        >
                          {task.status}
                        </span>
                      </td>
                      <td style={{ padding: '12px' }}>{task.user_id}</td>
                      <td style={{ padding: '12px' }}>{task.assignee || '-'}</td>
                      <td style={{ padding: '12px' }}>{(task.attempt_number || 0) + 1}</td>
                      <td style={{ padding: '12px' }}>{new Date(task.created_at).toLocaleString()}</td>
                      <td style={{ padding: '12px' }}>
                        {task.output_log_path ? (
                          <a
                            href={repo.getTaskLogUrl(task.id, 'output')}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            style={{ color: '#007bff', textDecoration: 'none' }}
                          >
                            View
                          </a>
                        ) : (
                          '-'
                        )}
                      </td>
                    </tr>
                    {isExpanded && hasMultipleAttempts && (
                      <tr>
                        <td colSpan={8} style={{ padding: '15px', backgroundColor: '#f8f9fa' }}>
                          <h4 style={{ marginTop: 0, marginBottom: '10px' }}>Attempt History</h4>
                          {loadingAttempts.has(task.id) ? (
                            <div>Loading attempts...</div>
                          ) : (
                            <table style={{ width: '100%', fontSize: '13px' }}>
                              <thead>
                                <tr style={{ borderBottom: '1px solid #dee2e6' }}>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Attempt</th>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Status</th>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Assignee</th>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Assigned</th>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Started</th>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Finished</th>
                                  <th style={{ padding: '8px', textAlign: 'left' }}>Logs</th>
                                </tr>
                              </thead>
                              <tbody>
                                {attempts.map((attempt) => (
                                  <tr key={attempt.id} style={{ borderBottom: '1px solid #e0e0e0' }}>
                                    <td style={{ padding: '8px' }}>{attempt.attempt_number + 1}</td>
                                    <td style={{ padding: '8px' }}>
                                      <span
                                        style={{
                                          padding: '2px 6px',
                                          borderRadius: '3px',
                                          backgroundColor: getStatusColor(attempt.status),
                                          color: 'white',
                                          fontSize: '11px',
                                        }}
                                      >
                                        {attempt.status}
                                      </span>
                                    </td>
                                    <td style={{ padding: '8px' }}>{attempt.assignee || '-'}</td>
                                    <td style={{ padding: '8px' }}>
                                      {attempt.assigned_at ? new Date(attempt.assigned_at).toLocaleString() : '-'}
                                    </td>
                                    <td style={{ padding: '8px' }}>
                                      {attempt.started_at ? new Date(attempt.started_at).toLocaleString() : '-'}
                                    </td>
                                    <td style={{ padding: '8px' }}>
                                      {attempt.finished_at ? new Date(attempt.finished_at).toLocaleString() : '-'}
                                    </td>
                                    <td style={{ padding: '8px' }}>
                                      {attempt.output_log_path ? (
                                        <a
                                          href={repo.getTaskLogUrl(task.id, 'output')}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          style={{ color: '#007bff', textDecoration: 'none' }}
                                        >
                                          View
                                        </a>
                                      ) : (
                                        '-'
                                      )}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
            </tbody>
          </table>
          {tasks.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No tasks found</div>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', padding: '20px 0' }}>
            <button
              onClick={() => loadTasks(currentPage - 1)}
              disabled={currentPage === 1}
              style={{
                padding: '8px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                backgroundColor: currentPage === 1 ? '#f3f4f6' : '#fff',
                cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
              }}
            >
              Previous
            </button>
            <span style={{ padding: '8px 12px' }}>
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => loadTasks(currentPage + 1)}
              disabled={currentPage === totalPages}
              style={{
                padding: '8px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                backgroundColor: currentPage === totalPages ? '#f3f4f6' : '#fff',
                cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
              }}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
