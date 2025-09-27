import React from 'react'
import { EvalTask } from '../repo'
import { AttributesRenderer } from './AttributesRenderer'
import {
  getDisplayStatus,
  getWorkingDuration,
  getStatusColor,
  getWorkerColor,
  truncateWorkerName,
  getGithubUrl,
} from '../utils/evalTasksUtils'

interface TaskRowProps {
  task: EvalTask
  isActive: boolean
  isExpanded: boolean
  onToggleExpansion: (taskId: string) => void
  showAssignee?: boolean
  showRetries?: boolean
  showUpdated?: boolean
}

export const TaskRow: React.FC<TaskRowProps> = ({
  task,
  isActive,
  isExpanded,
  onToggleExpansion,
  showAssignee = true,
  showRetries = true,
  showUpdated = true,
}) => {
  const displayStatus = getDisplayStatus(task)
  const isInProgress = displayStatus === 'in progress'
  const workingDuration = getWorkingDuration(task)
  const gitHash = task.attributes?.git_hash

  return (
    <React.Fragment>
      <tr
        style={{
          borderBottom: '1px solid #dee2e6',
          cursor: 'pointer',
          transition: 'background-color 0.2s',
        }}
        onClick={() => onToggleExpansion(task.id)}
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
        {isActive && (
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
        )}
        <td style={{ padding: '12px' }}>{task.user_id || '-'}</td>
        {showAssignee && (
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
        )}
        {showRetries && <td style={{ padding: '12px' }}>{task.retries}</td>}
        <td style={{ padding: '12px' }}>{new Date(task.created_at + 'Z').toLocaleString()}</td>
        {showUpdated && <td style={{ padding: '12px' }}>{new Date(task.updated_at + 'Z').toLocaleString()}</td>}
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={isActive ? (showAssignee ? 9 : 8) : showUpdated ? 6 : 5} style={{ padding: 0 }}>
            <AttributesRenderer attributes={task.attributes} />
          </td>
        </tr>
      )}
    </React.Fragment>
  )
}
