import clsx from 'clsx'
import { FC } from 'react'

import { EvalTask, TaskAttempt, TaskStatus } from '../repo'
import { Tooltip } from './Tooltip'

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

export const TaskBadge: FC<{ task: EvalTask | TaskAttempt; size?: 'small' | 'medium' }> = ({
  task,
  size = 'medium',
}) => {
  const errorReason = task.status_details?.error_reason
  const result = (
    <span
      className={clsx(
        size === 'small' ? 'text-xs py-0.5 px-1.5 rounded-sm' : 'text-xs px-2 py-1 rounded-[3px]',
        'text-white',
        errorReason && 'cursor-pointer'
      )}
      style={{
        backgroundColor: getStatusColor(task.status),
      }}
    >
      {task.status}
    </span>
  )
  if (errorReason) {
    return <Tooltip render={() => <div className="text-xs max-w-md">{errorReason}</div>}>{result}</Tooltip>
  } else {
    return result
  }
}
