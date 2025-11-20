import { FC } from 'react'

import { TaskStatus } from '../repo'

function getStatusColor(status: TaskStatus) {
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

export const TaskBadge: FC<{ status: TaskStatus; size?: 'small' | 'medium' }> = ({ status, size = 'medium' }) => {
  return (
    <span
      style={{
        padding: size === 'small' ? '2px 6px' : '4px 8px',
        borderRadius: size === 'small' ? '3px' : '4px',
        backgroundColor: getStatusColor(status),
        color: 'white',
        fontSize: size === 'small' ? '11px' : '12px',
      }}
    >
      {status}
    </span>
  )
}
