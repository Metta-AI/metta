import React, { useState } from 'react'
import { EvalTask } from '../repo'
import { SortField, SortDirection } from '../types/evalTasks'
import { SortHeader } from './SortHeader'
import { TaskRow } from './TaskRow'

interface TaskTableProps {
  tasks: EvalTask[]
  isActive: boolean
  expandedRows: Set<string>
  onToggleExpansion: (taskId: string) => void
  onSort: (field: SortField, isActive: boolean) => void
  activeSortField: SortField
  activeSortDirection: SortDirection
  completedSortField: SortField
  completedSortDirection: SortDirection
}

export const TaskTable: React.FC<TaskTableProps> = ({
  tasks,
  isActive,
  expandedRows,
  onToggleExpansion,
  onSort,
  activeSortField,
  activeSortDirection,
  completedSortField,
  completedSortDirection,
}) => {
  const [isTableExpanded, setIsTableExpanded] = useState(false)

  const getSortProps = () => ({
    onSort,
    activeSortField,
    activeSortDirection,
    completedSortField,
    completedSortDirection,
  })

  const displayTasks = isTableExpanded ? tasks : tasks.slice(0, 5)
  const hasMoreTasks = tasks.length > 5

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ backgroundColor: '#f8f9fa' }}>
            {isActive ? (
              <>
                <SortHeader field="policy_name" label="Policy" isActive={true} width="25%" {...getSortProps()} />
                <SortHeader field="sim_suite" label="Suite" isActive={true} width="8%" {...getSortProps()} />
                <SortHeader field="status" label="Status" isActive={true} width="12%" {...getSortProps()} />
                <SortHeader field="user_id" label="User" isActive={true} width="10%" {...getSortProps()} />
                <SortHeader field="assignee" label="Assignee" isActive={true} width="10%" {...getSortProps()} />
                <SortHeader field="retries" label="Tries" isActive={true} width="7%" {...getSortProps()} />
                <SortHeader field="created_at" label="Created" isActive={true} width="10%" {...getSortProps()} />
                <SortHeader field="updated_at" label="Updated" isActive={true} width="10%" {...getSortProps()} />
              </>
            ) : (
              <>
                <SortHeader field="policy_name" label="Policy" isActive={false} width="20%" {...getSortProps()} />
                <SortHeader field="sim_suite" label="Suite" isActive={false} width="10%" {...getSortProps()} />
                <SortHeader field="user_id" label="User" isActive={false} width="12%" {...getSortProps()} />
                <SortHeader field="status" label="Status" isActive={false} width="15%" {...getSortProps()} />
                <SortHeader field="created_at" label="Created" isActive={false} width="19%" {...getSortProps()} />
                <SortHeader field="updated_at" label="Updated" isActive={false} width="19%" {...getSortProps()} />
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {displayTasks.map((task) => (
            <TaskRow
              key={task.id}
              task={task}
              isActive={isActive}
              isExpanded={expandedRows.has(task.id)}
              onToggleExpansion={onToggleExpansion}
              showAssignee={isActive}
              showRetries={isActive}
              showUpdated={true}
            />
          ))}
        </tbody>
      </table>

      {tasks.length === 0 && (
        <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>
          {isActive ? 'No active tasks' : 'No task history'}
        </div>
      )}

      {hasMoreTasks && (
        <div
          style={{
            padding: '16px',
            textAlign: 'center',
            borderTop: '1px solid #e8e8e8',
            backgroundColor: '#f8f9fa',
          }}
        >
          <button
            onClick={() => setIsTableExpanded(!isTableExpanded)}
            style={{
              background: 'none',
              border: '1px solid #007bff',
              color: '#007bff',
              padding: '8px 16px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              margin: '0 auto',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#007bff'
              e.currentTarget.style.color = 'white'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent'
              e.currentTarget.style.color = '#007bff'
            }}
          >
            <span
              style={{
                transform: isTableExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease',
              }}
            >
              ▼
            </span>
            {isTableExpanded ? `Show Less (${tasks.length} total)` : `Show More (${tasks.length - 5} more)`}
          </button>
        </div>
      )}
    </div>
  )
}
