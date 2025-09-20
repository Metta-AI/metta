import React from 'react'
import { SortHeaderProps } from '../types/evalTasks'

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
