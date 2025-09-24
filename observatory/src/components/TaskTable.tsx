import React from 'react'
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
  // Pagination props
  currentPage: number
  totalCount: number
  pageSize: number
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
  currentPage,
  totalCount,
  pageSize,
}) => {
  const getSortProps = () => ({
    onSort,
    activeSortField,
    activeSortDirection,
    completedSortField,
    completedSortDirection,
  })

  // Calculate the range of items being displayed
  const startItem = (currentPage - 1) * pageSize + 1
  const endItem = Math.min(currentPage * pageSize, totalCount)

  return (
    <div>
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
            {tasks.map((task) => (
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
      </div>

      {/* Pagination info */}
      {tasks.length > 0 && (
        <div
          style={{
            padding: '16px',
            textAlign: 'center',
            borderTop: '1px solid #e8e8e8',
            backgroundColor: '#f8f9fa',
            color: '#6c757d',
            fontSize: '14px',
          }}
        >
          Showing {startItem}-{endItem} of {totalCount} tasks
        </div>
      )}
    </div>
  )
}
