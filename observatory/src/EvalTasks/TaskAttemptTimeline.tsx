import clsx from 'clsx'
import { FC } from 'react'

import { EvalTask, TaskAttempt } from '../repo'
import { formatDate, formatDurationBetween } from '../utils/datetime'

const getTimeDiffColor = (from: string | null, to: string | null): string => {
  if (!from || !to) return ''
  const fromTs = new Date(from).getTime()
  const toTs = new Date(to).getTime()
  const minutes = Math.floor((toTs - fromTs) / 1000 / 60)
  if (minutes < 2) return 'text-green-600'
  if (minutes < 10) return 'text-yellow-600'
  return 'text-red-500'
}

export const TaskAttemptTimeline: FC<{ task: EvalTask; attempt: TaskAttempt }> = ({ task, attempt }) => {
  const assignedDiff = formatDurationBetween(task.created_at, attempt.assigned_at)
  const startedDiff = formatDurationBetween(attempt.assigned_at, attempt.started_at)
  const finishedDiff = formatDurationBetween(attempt.started_at, attempt.finished_at)

  return (
    <div className="inline-grid grid-cols-[auto_auto_auto] gap-x-2 gap-y-1 text-xs">
      <div className="contents">
        <div className="text-gray-600">Assigned:</div>
        <div>{formatDate(attempt.assigned_at)}</div>
        <div className={clsx('text-right', getTimeDiffColor(task.created_at, attempt.assigned_at))}>
          {assignedDiff ? `(+ ${assignedDiff})` : ''}
        </div>
      </div>
      <div className="contents">
        <div className="text-gray-600">Started:</div>
        <div>{formatDate(attempt.started_at)}</div>
        <div className={clsx('text-right', getTimeDiffColor(attempt.assigned_at, attempt.started_at))}>
          {startedDiff ? `(+ ${startedDiff})` : ''}
        </div>
      </div>
      <div className="contents">
        <div className="text-gray-600">Finished:</div>
        <div>{formatDate(attempt.finished_at)}</div>
        <div className={clsx('text-right', getTimeDiffColor(attempt.started_at, attempt.finished_at))}>
          {finishedDiff ? `(+ ${finishedDiff})` : ''}
        </div>
      </div>
    </div>
  )
}
