import React, { useState, useEffect, useCallback } from 'react'
import { Repo, EvalTask, TaskFilters } from './repo'
import { METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO } from './constants'

interface TypeaheadInputProps {
  value: string
  onChange: (value: string) => void
  placeholder: string
  suggestions: string[]
  maxSuggestions?: number
  filterType?: 'prefix' | 'substring'
}

function TypeaheadInput({
  value,
  onChange,
  placeholder,
  suggestions,
  maxSuggestions = 10,
  filterType = 'substring',
}: TypeaheadInputProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])

  const handleInputChange = (inputValue: string) => {
    onChange(inputValue)
    if (inputValue.trim()) {
      const filtered = suggestions.filter((suggestion) => {
        const lowerSuggestion = suggestion.toLowerCase()
        const lowerInput = inputValue.toLowerCase()
        return filterType === 'prefix' ? lowerSuggestion.startsWith(lowerInput) : lowerSuggestion.includes(lowerInput)
      })
      setFilteredSuggestions(filtered.slice(0, maxSuggestions))
      setShowSuggestions(filtered.length > 0)
    } else {
      setShowSuggestions(false)
    }
  }

  return (
    <div style={{ position: 'relative' }}>
      <input
        type="text"
        value={value}
        onChange={(e) => handleInputChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%',
          padding: '10px 12px',
          borderRadius: '6px',
          border: '1px solid #d1d5db',
          fontSize: '14px',
          backgroundColor: '#fff',
          transition: 'border-color 0.2s',
          outline: 'none',
        }}
        onFocus={(e) => {
          e.target.style.borderColor = '#007bff'
          if (value.trim() && filteredSuggestions.length > 0) {
            setShowSuggestions(true)
          }
        }}
        onBlur={(e) => {
          e.target.style.borderColor = '#d1d5db'
          // Delay to allow clicking on suggestions
          setTimeout(() => setShowSuggestions(false), 200)
        }}
      />
      {showSuggestions && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            marginTop: '4px',
            backgroundColor: 'white',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            maxHeight: '200px',
            overflowY: 'auto',
            zIndex: 1000,
          }}
        >
          {filteredSuggestions.map((suggestion) => (
            <div
              key={suggestion}
              onClick={() => {
                onChange(suggestion)
                setShowSuggestions(false)
              }}
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                fontSize: '14px',
                borderBottom: '1px solid #f0f0f0',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f8f9fa')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'white')}
            >
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

interface PaginationProps {
  currentPage: number
  totalPages: number
  onPageChange: (page: number) => void
}

function Pagination({ currentPage, totalPages, onPageChange }: PaginationProps) {
  const getPageNumbers = () => {
    const pages: (number | string)[] = []
    const maxVisible = 7

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      pages.push(1)

      if (currentPage > 3) {
        pages.push('...')
      }

      const start = Math.max(2, currentPage - 1)
      const end = Math.min(totalPages - 1, currentPage + 1)

      for (let i = start; i <= end; i++) {
        pages.push(i)
      }

      if (currentPage < totalPages - 2) {
        pages.push('...')
      }

      pages.push(totalPages)
    }

    return pages
  }

  return (
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', justifyContent: 'center', padding: '20px 0' }}>
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        style={{
          padding: '8px 12px',
          border: '1px solid #d1d5db',
          borderRadius: '4px',
          backgroundColor: currentPage === 1 ? '#f3f4f6' : '#fff',
          cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
          fontSize: '14px',
        }}
      >
        Previous
      </button>

      {getPageNumbers().map((page, index) => {
        if (page === '...') {
          return (
            <span key={`ellipsis-${index}`} style={{ padding: '0 4px' }}>
              ...
            </span>
          )
        }

        const pageNum = page as number
        return (
          <button
            key={pageNum}
            onClick={() => onPageChange(pageNum)}
            style={{
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: '4px',
              backgroundColor: currentPage === pageNum ? '#007bff' : '#fff',
              color: currentPage === pageNum ? '#fff' : '#000',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: currentPage === pageNum ? 600 : 400,
            }}
          >
            {pageNum}
          </button>
        )
      })}

      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        style={{
          padding: '8px 12px',
          border: '1px solid #d1d5db',
          borderRadius: '4px',
          backgroundColor: currentPage === totalPages ? '#f3f4f6' : '#fff',
          cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
          fontSize: '14px',
        }}
      >
        Next
      </button>
    </div>
  )
}

type SortField =
  | 'policy_name'
  | 'sim_suite'
  | 'status'
  | 'assignee'
  | 'user_id'
  | 'retries'
  | 'created_at'
  | 'assigned_at'
  | 'updated_at'
type SortDirection = 'asc' | 'desc'

interface Props {
  repo: Repo
}

export function EvalTasks({ repo }: Props) {
  // Active tasks state
  const [activeTasks, setActiveTasks] = useState<EvalTask[]>([])
  const [activeCurrentPage, setActiveCurrentPage] = useState(1)
  const [activeTotalPages, setActiveTotalPages] = useState(1)
  const [activeTotalCount, setActiveTotalCount] = useState(0)
  const [activeFilters, setActiveFilters] = useState<TaskFilters>({ status: 'unprocessed' })

  // History tasks state
  const [historyTasks, setHistoryTasks] = useState<EvalTask[]>([])
  const [historyCurrentPage, setHistoryCurrentPage] = useState(1)
  const [historyTotalPages, setHistoryTotalPages] = useState(1)
  const [historyTotalCount, setHistoryTotalCount] = useState(0)
  const [historyFilters, setHistoryFilters] = useState<TaskFilters>({})

  // Form state
  const [policyIdInput, setPolicyIdInput] = useState<string>('')
  const [gitHash, setGitHash] = useState<string>('')
  const [simSuite, setSimSuite] = useState<string>('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // UI state
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())
  const [recentPolicies, setRecentPolicies] = useState<string[]>([])
  const [recentSimSuites, setRecentSimSuites] = useState<string[]>([])

  // Track if this is the first render to avoid duplicate initial loads
  const isFirstRender = React.useRef(true)

  const pageSize = 50

  // Load active tasks with filters
  const loadActiveTasks = async (page: number) => {
    try {
      console.log('Loading active tasks with filters:', activeFilters)
      const response = await repo.getEvalTasksPaginated(page, pageSize, activeFilters)
      console.log('Active tasks response:', { count: response.tasks.length, total: response.total_count })
      setActiveTasks(response.tasks)
      setActiveCurrentPage(response.page)
      setActiveTotalPages(response.total_pages)
      setActiveTotalCount(response.total_count)

      // Update suggestions from loaded tasks
      updateSuggestionsFromTasks(response.tasks)
    } catch (err: any) {
      console.error('Failed to load active tasks:', err)
    }
  }

  // Load history tasks with filters
  const loadHistoryTasks = async (page: number) => {
    try {
      console.log('Loading history tasks with filters:', historyFilters)
      const response = await repo.getEvalTasksPaginated(page, pageSize, historyFilters)
      console.log('History tasks response:', { count: response.tasks.length, total: response.total_count })
      setHistoryTasks(response.tasks)
      setHistoryCurrentPage(response.page)
      setHistoryTotalPages(response.total_pages)
      setHistoryTotalCount(response.total_count)

      // Update suggestions from loaded tasks
      updateSuggestionsFromTasks(response.tasks)
    } catch (err: any) {
      console.error('Failed to load history tasks:', err)
    }
  }

  // Update suggestions from tasks
  const updateSuggestionsFromTasks = (tasks: EvalTask[]) => {
    setRecentPolicies((prev) => {
      const policySet = new Set(prev)
      tasks.forEach((task) => {
        if (task.policy_name) policySet.add(task.policy_name)
        policySet.add(task.policy_id)
      })
      return Array.from(policySet).sort()
    })

    setRecentSimSuites((prev) => {
      const simSuiteSet = new Set(prev)
      tasks.forEach((task) => {
        if (task.sim_suite) simSuiteSet.add(task.sim_suite)
      })
      return Array.from(simSuiteSet).sort()
    })
  }

  // Initial load
  useEffect(() => {
    loadActiveTasks(1)
    loadHistoryTasks(1)
    isFirstRender.current = false
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Reload when filters change (with debouncing)
  useEffect(() => {
    if (isFirstRender.current) return

    console.log('Active filters changed, will reload in 300ms:', activeFilters)
    const timeoutId = setTimeout(() => {
      loadActiveTasks(1)
    }, 300) // 300ms debounce

    return () => clearTimeout(timeoutId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeFilters])

  useEffect(() => {
    if (isFirstRender.current) return

    console.log('History filters changed, will reload in 300ms:', historyFilters)
    const timeoutId = setTimeout(() => {
      loadHistoryTasks(1)
    }, 300) // 300ms debounce

    return () => clearTimeout(timeoutId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historyFilters])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadActiveTasks(activeCurrentPage)
      loadHistoryTasks(historyCurrentPage)
    }, 5000)

    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCurrentPage, historyCurrentPage, activeFilters, historyFilters])

  const handleCreateTask = async () => {
    if (!policyIdInput.trim()) {
      setError('Please enter a policy name or ID')
      return
    }

    setLoading(true)
    setError(null)

    try {
      let policyId = policyIdInput.trim()

      // Check if input looks like a UUID
      const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(policyId)

      if (!isUuid) {
        // Try to resolve policy name to ID
        try {
          const policyIds = await repo.getPolicyIds([policyId])
          if (policyIds[policyId]) {
            policyId = policyIds[policyId]
          } else {
            setError(`Policy with name '${policyId}' not found`)
            return
          }
        } catch (e) {
          setError(`Failed to resolve policy name: ${e}`)
          return
        }
      }

      await repo.createEvalTask({
        policy_id: policyId,
        git_hash: gitHash || null,
        sim_suite: simSuite,
        env_overrides: {},
      })

      // Clear form
      setPolicyIdInput('')
      setGitHash('')

      // Refresh tasks
      await loadActiveTasks(1)
    } catch (err: any) {
      setError(`Failed to create task: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string, isInProgress: boolean) => {
    if (isInProgress) return '#17a2b8'
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

  const getDisplayStatus = (task: EvalTask) => {
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

  const getWorkingDuration = (task: EvalTask) => {
    if (getDisplayStatus(task) !== 'in progress') return null
    if (!task.assigned_at) return null

    const start = new Date(task.assigned_at + 'Z')
    const now = new Date()
    const diff = now.getTime() - start.getTime()

    if (diff < 0) return '00:00'

    const totalSeconds = Math.floor(diff / 1000)
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = totalSeconds % 60

    const formattedMinutes = minutes.toString().padStart(2, '0')
    const formattedSeconds = seconds.toString().padStart(2, '0')

    return `${formattedMinutes}:${formattedSeconds}`
  }

  const getGithubUrl = (gitHash: string) => {
    return `https://github.com/${METTA_GITHUB_ORGANIZATION}/${METTA_GITHUB_REPO}/commit/${gitHash}`
  }

  const truncateWorkerName = (workerName: string | null) => {
    if (!workerName) return ''
    const parts = workerName.split('-')
    if (parts.length >= 3) {
      const suffix = parts[parts.length - 1]
      return suffix
    }
    return workerName
  }

  const getWorkerColor = (workerName: string | null) => {
    if (!workerName) return 'transparent'

    let hash = 0
    for (let i = 0; i < workerName.length; i++) {
      hash = workerName.charCodeAt(i) + ((hash << 5) - hash)
    }

    const hue = hash % 360
    return `hsl(${hue}, 70%, 85%)`
  }

  const toggleRowExpansion = (taskId: string) => {
    const newExpanded = new Set(expandedRows)
    if (newExpanded.has(taskId)) {
      newExpanded.delete(taskId)
    } else {
      newExpanded.add(taskId)
    }
    setExpandedRows(newExpanded)
  }

  const renderAttributes = (attributes: Record<string, any>) => {
    if (!attributes || Object.keys(attributes).length === 0) {
      return <div style={{ padding: '10px', color: '#6c757d' }}>No attributes</div>
    }

    const formatValue = (value: any): React.ReactNode => {
      if (typeof value === 'string') {
        const lines = value.split('\\n')
        if (lines.length > 1) {
          return (
            <>
              {lines.map((line, i) => (
                <React.Fragment key={i}>
                  {i > 0 && <br />}
                  {line}
                </React.Fragment>
              ))}
            </>
          )
        }
        return value
      }
      return JSON.stringify(value, null, 2)
    }

    const renderObject = (obj: Record<string, any>, indent: number = 0): React.ReactNode => {
      const entries = Object.entries(obj).filter(([_, value]) => {
        if (value === null || value === undefined || value === '' || value === false) return false
        if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0) return false
        if (Array.isArray(value) && value.length === 0) return false
        return true
      })

      return (
        <div style={{ marginLeft: indent > 0 ? '20px' : 0 }}>
          {entries.map(([key, value], i) => (
            <div key={key} style={{ marginBottom: i < entries.length - 1 ? '8px' : 0 }}>
              <span style={{ color: '#0066cc', fontWeight: 500 }}>{key}:</span>{' '}
              {typeof value === 'object' && value !== null && !Array.isArray(value) ? (
                renderObject(value, indent + 1)
              ) : (
                <span style={{ color: '#333' }}>{formatValue(value)}</span>
              )}
            </div>
          ))}
        </div>
      )
    }

    return (
      <div
        style={{
          padding: '15px',
          backgroundColor: '#f8f9fa',
          borderTop: '1px solid #dee2e6',
        }}
      >
        <h4
          style={{
            marginTop: 0,
            marginBottom: '10px',
            fontSize: '14px',
            fontWeight: 600,
          }}
        >
          Attributes
        </h4>
        <div
          style={{
            fontSize: '12px',
            lineHeight: 1.6,
            fontFamily: 'Monaco, Consolas, "Courier New", monospace',
          }}
        >
          {renderObject(attributes)}
        </div>
      </div>
    )
  }

  const renderFilterInput = (
    value: string,
    onChange: (value: string) => void,
    placeholder: string = 'Filter...'
  ) => {
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
        onChange={(e) => {
          console.log('Status dropdown changed to:', e.target.value)
          onChange(e.target.value)
        }}
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
        <option value="done">Done</option>
        <option value="error">Error</option>
        <option value="canceled">Canceled</option>
      </select>
    )
  }

  const renderActiveTasksTable = () => {
    return (
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ marginBottom: '20px' }}>Active ({activeTotalCount})</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa' }}>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '25%' }}>
                  Policy
                  {renderFilterInput(activeFilters.policy_name || '', (value) =>
                    setActiveFilters({ ...activeFilters, policy_name: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '10%' }}>
                  Suite
                  {renderFilterInput(activeFilters.sim_suite || '', (value) =>
                    setActiveFilters({ ...activeFilters, sim_suite: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                  User
                  {renderFilterInput(activeFilters.user_id || '', (value) =>
                    setActiveFilters({ ...activeFilters, user_id: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '15%' }}>
                  Assignee
                  {renderFilterInput(activeFilters.assignee || '', (value) =>
                    setActiveFilters({ ...activeFilters, assignee: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '8%' }}>
                  Tries
                  {renderFilterInput(activeFilters.retries || '', (value) =>
                    setActiveFilters({ ...activeFilters, retries: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                  Created
                  {renderFilterInput(activeFilters.created_at || '', (value) =>
                    setActiveFilters({ ...activeFilters, created_at: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                  Updated
                  {renderFilterInput(activeFilters.updated_at || '', (value) =>
                    setActiveFilters({ ...activeFilters, updated_at: value })
                  )}
                </th>
              </tr>
            </thead>
            <tbody>
              {activeTasks.map((task) => {
                const displayStatus = getDisplayStatus(task)
                const isInProgress = displayStatus === 'in progress'
                const workingDuration = getWorkingDuration(task)
                const gitHash = task.attributes?.git_hash
                const isExpanded = expandedRows.has(task.id)

                return (
                  <React.Fragment key={task.id}>
                    <tr
                      style={{
                        borderBottom: '1px solid #dee2e6',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s',
                      }}
                      onClick={() => toggleRowExpansion(task.id)}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f8f9fa')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '')}
                    >
                      <td style={{ padding: '12px', position: 'relative' }}>
                        <span
                          style={{
                            position: 'absolute',
                            left: '12px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            fontSize: '12px',
                            color: '#6c757d',
                          }}
                        >
                          {isExpanded ? '▼' : '▶'}
                        </span>
                        <div style={{ paddingLeft: '20px' }}>
                          <span>{task.policy_name || task.policy_id}</span>
                          {gitHash && (
                            <div style={{ fontSize: '12px', marginTop: '2px' }}>
                              <a
                                href={getGithubUrl(gitHash)}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{
                                  color: '#6c757d',
                                  textDecoration: 'none',
                                }}
                                onClick={(e) => e.stopPropagation()}
                              >
                                {gitHash.substring(0, 8)}
                              </a>
                            </div>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>{task.sim_suite}</td>
                      <td style={{ padding: '12px' }}>{task.user_id || '-'}</td>
                      <td style={{ padding: '12px' }}>
                        <span
                          style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            backgroundColor: getWorkerColor(task.assignee),
                            fontSize: '12px',
                            display: 'inline-block',
                          }}
                        >
                          {truncateWorkerName(task.assignee)}
                        </span>
                        <span>{workingDuration || ''}</span>
                      </td>
                      <td style={{ padding: '12px' }}>{task.retries}</td>
                      <td style={{ padding: '12px' }}>{new Date(task.created_at + 'Z').toLocaleString()}</td>
                      <td style={{ padding: '12px' }}>{new Date(task.updated_at + 'Z').toLocaleString()}</td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <td colSpan={7} style={{ padding: 0 }}>
                          {renderAttributes(task.attributes)}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
            </tbody>
          </table>
          {activeTasks.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No active tasks</div>
          )}
        </div>
        {activeTotalPages > 1 && (
          <Pagination
            currentPage={activeCurrentPage}
            totalPages={activeTotalPages}
            onPageChange={(page) => {
              setActiveCurrentPage(page)
              loadActiveTasks(page)
            }}
          />
        )}
      </div>
    )
  }

  const renderHistoryTasksTable = () => {
    return (
      <div>
        <h2 style={{ marginBottom: '20px' }}>History ({historyTotalCount})</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa' }}>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '20%' }}>
                  Policy
                  {renderFilterInput(historyFilters.policy_name || '', (value) =>
                    setHistoryFilters({ ...historyFilters, policy_name: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '10%' }}>
                  Suite
                  {renderFilterInput(historyFilters.sim_suite || '', (value) =>
                    setHistoryFilters({ ...historyFilters, sim_suite: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '12%' }}>
                  User
                  {renderFilterInput(historyFilters.user_id || '', (value) =>
                    setHistoryFilters({ ...historyFilters, user_id: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '15%' }}>
                  Status
                  {renderStatusDropdown(historyFilters.status || '', (value) =>
                    setHistoryFilters({ ...historyFilters, status: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '19%' }}>
                  Created
                  {renderFilterInput(historyFilters.created_at || '', (value) =>
                    setHistoryFilters({ ...historyFilters, created_at: value })
                  )}
                </th>
                <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #dee2e6', width: '19%' }}>
                  Updated
                  {renderFilterInput(historyFilters.updated_at || '', (value) =>
                    setHistoryFilters({ ...historyFilters, updated_at: value })
                  )}
                </th>
              </tr>
            </thead>
            <tbody>
              {historyTasks.map((task) => {
                const gitHash = task.attributes?.git_hash
                const isExpanded = expandedRows.has(task.id)

                return (
                  <React.Fragment key={task.id}>
                    <tr
                      style={{
                        borderBottom: '1px solid #dee2e6',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s',
                      }}
                      onClick={() => toggleRowExpansion(task.id)}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f8f9fa')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '')}
                    >
                      <td style={{ padding: '12px', position: 'relative' }}>
                        <span
                          style={{
                            position: 'absolute',
                            left: '12px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            fontSize: '12px',
                            color: '#6c757d',
                          }}
                        >
                          {isExpanded ? '▼' : '▶'}
                        </span>
                        <div style={{ paddingLeft: '20px' }}>
                          <span>{task.policy_name || task.policy_id}</span>
                          {gitHash && (
                            <div style={{ fontSize: '12px', marginTop: '2px' }}>
                              <a
                                href={getGithubUrl(gitHash)}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{
                                  color: '#6c757d',
                                  textDecoration: 'none',
                                }}
                                onClick={(e) => e.stopPropagation()}
                              >
                                {gitHash.substring(0, 8)}
                              </a>
                            </div>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>{task.sim_suite}</td>
                      <td style={{ padding: '12px' }}>{task.user_id || '-'}</td>
                      <td style={{ padding: '12px' }}>
                        <div>
                          <span
                            style={{
                              padding: '4px 8px',
                              borderRadius: '4px',
                              backgroundColor: getStatusColor(task.status, false),
                              color: 'white',
                              fontSize: '12px',
                            }}
                          >
                            {task.status}
                          </span>
                          {task.status === 'error' && task.attributes?.details?.error && (
                            <div
                              style={{
                                fontSize: '11px',
                                color: '#dc3545',
                                marginTop: '4px',
                                maxWidth: '200px',
                                wordBreak: 'break-word',
                              }}
                            >
                              {task.attributes.details.error}
                            </div>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>{new Date(task.created_at + 'Z').toLocaleString()}</td>
                      <td style={{ padding: '12px' }}>{new Date(task.updated_at + 'Z').toLocaleString()}</td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <td colSpan={6} style={{ padding: 0 }}>
                          {renderAttributes(task.attributes)}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
            </tbody>
          </table>
          {historyTasks.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No task history</div>
          )}
        </div>
        {historyTotalPages > 1 && (
          <Pagination
            currentPage={historyCurrentPage}
            totalPages={historyTotalPages}
            onPageChange={(page) => {
              setHistoryCurrentPage(page)
              loadHistoryTasks(page)
            }}
          />
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '30px' }}>Policy Evaluation Tasks</h1>

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

      {/* Policy Selection Section */}
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
        <h3
          style={{
            marginTop: 0,
            marginBottom: '20px',
            fontSize: '18px',
            fontWeight: 600,
            color: '#1a1a1a',
          }}
        >
          Create New Evaluation Task
        </h3>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end' }}>
          <div style={{ flex: '1 1 300px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '6px',
                fontSize: '13px',
                fontWeight: 500,
                color: '#555',
              }}
            >
              Policy Name or ID
            </label>
            <TypeaheadInput
              value={policyIdInput}
              onChange={setPolicyIdInput}
              placeholder="Enter policy name or ID"
              suggestions={recentPolicies}
              maxSuggestions={10}
              filterType="substring"
            />
          </div>

          <div style={{ flex: '1 1 250px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '6px',
                fontSize: '13px',
                fontWeight: 500,
                color: '#555',
              }}
            >
              Git Commit
            </label>
            <input
              type="text"
              value={gitHash}
              onChange={(e) => setGitHash(e.target.value)}
              placeholder="Latest main (default)"
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                fontSize: '14px',
                backgroundColor: '#fff',
                transition: 'border-color 0.2s',
                outline: 'none',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#007bff')}
              onBlur={(e) => (e.target.style.borderColor = '#d1d5db')}
            />
          </div>

          <div style={{ flex: '0 0 140px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '6px',
                fontSize: '13px',
                fontWeight: 500,
                color: '#555',
              }}
            >
              Suite
            </label>
            <TypeaheadInput
              value={simSuite}
              onChange={setSimSuite}
              placeholder="Enter sim suite"
              suggestions={recentSimSuites}
              maxSuggestions={5}
              filterType="prefix"
            />
          </div>

          <button
            onClick={handleCreateTask}
            disabled={loading || !policyIdInput.trim()}
            style={{
              padding: '10px 24px',
              backgroundColor: loading || !policyIdInput.trim() ? '#9ca3af' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !policyIdInput.trim() ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 500,
              transition: 'background-color 0.2s',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={(e) => {
              if (!loading && policyIdInput.trim()) {
                e.currentTarget.style.backgroundColor = '#0056b3'
              }
            }}
            onMouseLeave={(e) => {
              if (!loading && policyIdInput.trim()) {
                e.currentTarget.style.backgroundColor = '#007bff'
              }
            }}
          >
            {loading ? 'Creating...' : 'Create Task'}
          </button>
        </div>
      </div>

      {renderActiveTasksTable()}
      {renderHistoryTasksTable()}
    </div>
  )
}
