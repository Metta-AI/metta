import React, { useState, useEffect } from 'react'
import { Repo, EvalTask } from './repo'
import { METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO } from './constants'

interface TypeaheadInputProps {
  value: string
  onChange: (value: string) => void
  placeholder: string
  suggestions: string[]
  maxSuggestions?: number
  filterType?: 'prefix' | 'substring'
  grouped?: boolean
}

// --- replace InputGroup, GroupCell, GroupTextInput ---

function InputGroup({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        position: 'relative',
        display: 'grid',
        gridTemplateColumns: '1.5fr 1fr 180px 140px', // ⬅️ same as labels
        alignItems: 'stretch',
        border: '1px solid #d1d5db',
        borderRadius: 8,
        background: '#fff',
        overflow: 'visible', // allow dropdowns to escape
      }}
    >
      {children}
    </div>
  )
}

function GroupCell({
  children,
  withDivider = false,
  isFirst = false,
  isLast = false,
}: {
  children: React.ReactNode
  withDivider?: boolean
  isFirst?: boolean
  isLast?: boolean
}) {
  const [focused, setFocused] = React.useState(false)

  return (
    <div
      onFocusCapture={() => setFocused(true)}
      onBlurCapture={() => setFocused(false)}
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        height: 38,
        borderLeft: withDivider ? (focused ? '1px solid transparent' : '1px solid #e5e7eb') : 'none',
      }}
    >
      {/* focus ring drawn on the CELL, with correct corner rounding */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          inset: focused ? -1 : 0,
          pointerEvents: 'none',
          border: focused ? '2px solid #007bff' : '2px solid transparent',
          borderRadius: isFirst ? '8px 0 0 8px' : isLast ? '0 8px 8px 0' : 0,
          zIndex: 1,
        }}
      />
      <div style={{ position: 'relative', flex: 1, height: '100%' }}>{children}</div>
    </div>
  )
}

function GroupTextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      style={{
        width: '100%',
        height: '100%',
        padding: '0 12px',
        lineHeight: '38px',
        border: 'none',
        borderRadius: 0,
        fontSize: 14,
        background: 'transparent',
        outline: 'none', // cell shows focus
      }}
    />
  )
}

function TypeaheadInput({
  value,
  onChange,
  placeholder,
  suggestions,
  maxSuggestions = 10,
  filterType = 'substring',
  grouped = false,
}: TypeaheadInputProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])

  const handleInputChange = (inputValue: string) => {
    onChange(inputValue)
    if (inputValue.trim()) {
      const lower = inputValue.toLowerCase()
      const filtered = suggestions.filter((s) =>
        filterType === 'prefix' ? s.toLowerCase().startsWith(lower) : s.toLowerCase().includes(lower)
      )
      setFilteredSuggestions(filtered.slice(0, maxSuggestions))
      setShowSuggestions(filtered.length > 0)
    } else {
      setShowSuggestions(false)
    }
  }

  const baseInputStyle: React.CSSProperties = grouped
    ? {
        width: '100%',
        height: '100%',
        padding: '0 12px',
        lineHeight: '38px',
        border: 'none',
        borderRadius: 0,
        fontSize: 14,
        backgroundColor: 'transparent',
        outline: 'none', // cell shows focus
      }
    : {
        width: '100%',
        padding: '10px 12px',
        borderRadius: 6,
        border: '1px solid #d1d5db',
        fontSize: 14,
        backgroundColor: '#fff',
        outline: 'none',
      }

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <input
        type="text"
        value={value}
        onChange={(e) => handleInputChange(e.target.value)}
        placeholder={placeholder}
        style={baseInputStyle}
        // REMOVE the boxShadow/borderColor focus handlers
        onFocus={() => {
          if (value.trim() && filteredSuggestions.length > 0) setShowSuggestions(true)
        }}
        onBlur={() => {
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
            marginTop: 4,
            backgroundColor: 'white',
            border: '1px solid #d1d5db',
            borderRadius: 6,
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            maxHeight: 200,
            overflowY: 'auto',
            zIndex: 50, // above neighbors & the button
          }}
        >
          ...
        </div>
      )}
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
  const [tasks, setTasks] = useState<EvalTask[]>([])
  const [policyIdInput, setPolicyIdInput] = useState<string>('')
  const [gitHash, setGitHash] = useState<string>('')
  const [simSuite, setSimSuite] = useState<string>('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeSortField, setActiveSortField] = useState<SortField>('created_at')
  const [activeSortDirection, setActiveSortDirection] = useState<SortDirection>('desc')
  const [completedSortField, setCompletedSortField] = useState<SortField>('created_at')
  const [completedSortDirection, setCompletedSortDirection] = useState<SortDirection>('desc')
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())

  // Set up auto-refresh for tasks
  useEffect(() => {
    loadTasks()
    const interval = setInterval(loadTasks, 5000) // Refresh every 5 seconds

    return () => {
      clearInterval(interval)
    }
  }, [])

  const loadTasks = async () => {
    try {
      const tasks = await repo.getEvalTasks()
      setTasks(tasks)
    } catch (err: any) {
      console.error('Failed to refresh tasks:', err)
    }
  }

  // Get unique policy names/IDs from recent tasks for typeahead
  const getRecentPolicies = (): string[] => {
    const policySet = new Set<string>()
    tasks.forEach((task) => {
      if (task.policy_name) policySet.add(task.policy_name)
      policySet.add(task.policy_id)
    })
    return Array.from(policySet).sort()
  }

  // Get unique sim suite values from recent tasks for autocomplete
  const getRecentSimSuites = (): string[] => {
    const simSuiteSet = new Set<string>()
    tasks.forEach((task) => {
      if (task.sim_suite) simSuiteSet.add(task.sim_suite)
    })
    return Array.from(simSuiteSet).sort()
  }

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
      await loadTasks()
    } catch (err: any) {
      setError(`Failed to create task: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string, isInProgress: boolean) => {
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

  const getDisplayStatus = (task: EvalTask) => {
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

  const getWorkingDuration = (task: EvalTask) => {
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

  const getGithubUrl = (gitHash: string) => {
    // Assuming metta repo, adjust if needed
    return `https://github.com/${METTA_GITHUB_ORGANIZATION}/${METTA_GITHUB_REPO}/commit/${gitHash}`
  }

  const truncateWorkerName = (workerName: string | null) => {
    if (!workerName) return ''
    const parts = workerName.split('-')
    if (parts.length >= 3) {
      // Get the last part (suffix) and abbreviate the middle parts
      const suffix = parts[parts.length - 1]
      return suffix
    }
    return workerName
  }

  const getWorkerColor = (workerName: string | null) => {
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

  const sortTasks = (tasksToSort: EvalTask[], field: SortField, direction: SortDirection) => {
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

  const handleSort = (field: SortField, isActive: boolean) => {
    if (isActive) {
      setActiveSortField(field)
      setActiveSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setCompletedSortField(field)
      setCompletedSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    }
  }

  const SortHeader = ({
    field,
    label,
    isActive,
    width,
  }: {
    field: SortField
    label: string
    isActive: boolean
    width?: string
  }) => {
    const sortField = isActive ? activeSortField : completedSortField
    const sortDirection = isActive ? activeSortDirection : completedSortDirection
    const isCurrentSort = sortField === field

    return (
      <th
        style={{
          padding: '12px',
          textAlign: 'left',
          borderBottom: '2px solid #dee2e6',
          cursor: 'pointer',
          userSelect: 'none',
          position: 'relative',
          width: width,
        }}
        onClick={() => handleSort(field, isActive)}
      >
        {label}
        {isCurrentSort && (
          <span style={{ marginLeft: '5px', fontSize: '12px' }}>{sortDirection === 'asc' ? '▲' : '▼'}</span>
        )}
      </th>
    )
  }

  const activeTasks = tasks.filter((t) => t.status === 'unprocessed')
  const historyTasks = tasks.filter((t) => t.status !== 'unprocessed')

  const sortedActiveTasks = sortTasks(activeTasks, activeSortField, activeSortDirection)
  const sortedHistoryTasks = sortTasks(historyTasks, completedSortField, completedSortDirection)

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
        // Split by newlines and render each line separately
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
      // Filter out empty/falsy values
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
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1.5fr 1fr 180px 140px', // ⬅️ match InputGroup
            columnGap: 0, // ⬅️ no gap so columns line up perfectly
            marginBottom: 6,
          }}
        >
          <label style={{ padding: '0 12px', fontSize: 13, fontWeight: 500, color: '#555' }}>Policy Name or ID</label>
          <label style={{ padding: '0 12px', fontSize: 13, fontWeight: 500, color: '#555' }}>Git Commit</label>
          <label style={{ padding: '0 12px', fontSize: 13, fontWeight: 500, color: '#555' }}>Suite</label>
          <span />
        </div>
        <InputGroup>
          <GroupCell isFirst>
            <TypeaheadInput
              value={policyIdInput}
              onChange={setPolicyIdInput}
              placeholder="Enter policy name or ID"
              suggestions={getRecentPolicies()}
              maxSuggestions={10}
              filterType="substring"
              grouped
            />
          </GroupCell>

          <GroupCell withDivider>
            <GroupTextInput
              value={gitHash}
              onChange={(e) => setGitHash((e.target as HTMLInputElement).value)}
              placeholder="Latest main (default)"
            />
          </GroupCell>

          <GroupCell withDivider>
            <TypeaheadInput
              value={simSuite}
              onChange={setSimSuite}
              placeholder="Enter sim suite"
              suggestions={getRecentSimSuites()}
              maxSuggestions={5}
              filterType="prefix"
              grouped
            />
          </GroupCell>

          <GroupCell withDivider isLast>
            <button
              onClick={handleCreateTask}
              disabled={loading || !policyIdInput.trim()}
              style={{
                width: '100%',
                height: '100%',
                backgroundColor: loading || !policyIdInput.trim() ? '#9ca3af' : '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '0 8px 8px 0', // ⬅️ match outer corners
                cursor: loading || !policyIdInput.trim() ? 'not-allowed' : 'pointer',
                fontSize: 14,
                fontWeight: 500,
              }}
              onMouseEnter={(e) => {
                if (!loading && policyIdInput.trim()) e.currentTarget.style.backgroundColor = '#0056b3'
              }}
              onMouseLeave={(e) => {
                if (!loading && policyIdInput.trim()) e.currentTarget.style.backgroundColor = '#007bff'
              }}
            >
              {loading ? 'Creating...' : 'Create Task'}
            </button>
          </GroupCell>
        </InputGroup>
      </div>

      {/* Active Section */}
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ marginBottom: '20px' }}>Active ({activeTasks.length})</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa' }}>
                <SortHeader field="policy_name" label="Policy" isActive={true} width="25%" />
                <SortHeader field="sim_suite" label="Suite" isActive={true} width="8%" />
                <SortHeader field="status" label="Status" isActive={true} width="12%" />
                <SortHeader field="user_id" label="User" isActive={true} width="10%" />
                <SortHeader field="assignee" label="Assignee" isActive={true} width="10%" />
                <SortHeader field="retries" label="Tries" isActive={true} width="7%" />
                <SortHeader field="created_at" label="Created" isActive={true} width="10%" />
                <SortHeader field="updated_at" label="Updated" isActive={true} width="10%" />
              </tr>
            </thead>
            <tbody>
              {sortedActiveTasks.map((task) => {
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
                              >
                                {gitHash.substring(0, 8)}
                              </a>
                            </div>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>{task.sim_suite}</td>
                      <td style={{ padding: '12px' }}>
                        <div>
                          <span
                            style={{
                              padding: '4px 8px',
                              borderRadius: '4px',
                              backgroundColor: getStatusColor(task.status, isInProgress),
                              color: 'white',
                              fontSize: '12px',
                              fontWeight: isInProgress ? 600 : 400,
                              textTransform: isInProgress ? 'uppercase' : 'none',
                              letterSpacing: isInProgress ? '0.5px' : '0',
                            }}
                          >
                            {displayStatus}
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
                        <td colSpan={9} style={{ padding: 0 }}>
                          {renderAttributes(task.attributes)}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
            </tbody>
          </table>
          {sortedActiveTasks.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No active tasks</div>
          )}
        </div>
      </div>

      {/* History Section */}
      <div>
        <h2 style={{ marginBottom: '20px' }}>History ({historyTasks.length})</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa' }}>
                <SortHeader field="policy_name" label="Policy" isActive={false} width="20%" />
                <SortHeader field="sim_suite" label="Suite" isActive={false} width="10%" />
                <SortHeader field="user_id" label="User" isActive={false} width="12%" />
                <SortHeader field="status" label="Status" isActive={false} width="15%" />
                <SortHeader field="created_at" label="Created" isActive={false} width="19%" />
                <SortHeader field="updated_at" label="Updated" isActive={false} width="19%" />
              </tr>
            </thead>
            <tbody>
              {sortedHistoryTasks.map((task) => {
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
          {sortedHistoryTasks.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>No task history</div>
          )}
        </div>
      </div>
    </div>
  )
}
