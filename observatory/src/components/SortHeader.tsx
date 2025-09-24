import React from 'react'

export type SortField =
  | 'policy_name'
  | 'sim_suite'
  | 'status'
  | 'assignee'
  | 'user_id'
  | 'retries'
  | 'created_at'
  | 'assigned_at'
  | 'updated_at'

export type SortDirection = 'asc' | 'desc'

interface SortHeaderProps {
  field: SortField
  label: string
  isActive: boolean
  width?: string
  onSort: (field: SortField, isActive: boolean) => void
  activeSortField: SortField
  activeSortDirection: SortDirection
  completedSortField: SortField
  completedSortDirection: SortDirection
}

export const SortHeader: React.FC<SortHeaderProps> = ({
  field,
  label,
  isActive,
  width,
  onSort,
  activeSortField,
  activeSortDirection,
  completedSortField,
  completedSortDirection,
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
      onClick={() => onSort(field, isActive)}
    >
      {label}
      {isCurrentSort && (
        <span style={{ marginLeft: '5px', fontSize: '12px' }}>{sortDirection === 'asc' ? '▲' : '▼'}</span>
      )}
    </th>
  )
}
