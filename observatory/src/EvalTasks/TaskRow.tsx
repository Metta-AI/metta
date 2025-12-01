import clsx from 'clsx'
import { FC, Fragment, useState } from 'react'

import { TaskBadge } from '../components/TaskBadge'
import { EvalTask, Repo, TaskAttempt } from '../repo'

export const TaskRow: FC<{ task: EvalTask; repo: Repo }> = ({ task, repo }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  // UI state
  const [attempts, setAttempts] = useState<TaskAttempt[]>([])
  const [isLoadingAttempts, setIsLoadingAttempts] = useState(false)

  const hasMultipleAttempts = (task.attempt_number || 0) > 0

  // Toggle task expansion
  const toggleTaskExpansion = async () => {
    if (isExpanded) {
      setIsExpanded(false)
    } else {
      // Load attempts if not already loaded
      if (!isLoadingAttempts) {
        setIsLoadingAttempts(true)
        try {
          const response = await repo.getTaskAttempts(task.id)
          setAttempts(response.attempts)
        } catch (err) {
          console.error('Failed to load attempts:', err)
        } finally {
          setIsLoadingAttempts(false)
        }
      }
      setIsExpanded(true)
    }
  }

  return (
    <Fragment>
      <tr
        className={clsx('border-b border-gray-200', hasMultipleAttempts && 'cursor-pointer')}
        onClick={() => hasMultipleAttempts && toggleTaskExpansion()}
      >
        <td className="p-3">
          {hasMultipleAttempts && <span className="mr-2 text-xs text-gray-500">{isExpanded ? '▼' : '▶'}</span>}
          {task.id}
          {isLoadingAttempts && <span className="mr-2 text-xs text-gray-500">...</span>}
        </td>
        <td className="p-3 text-xs truncate text-wrap" title={task.command}>
          {task.command}
        </td>
        <td className="p-3">
          <TaskBadge task={task} />
        </td>
        <td className="p-3 text-sm truncate" title={task.user_id}>
          {task.user_id}
        </td>
        <td className="p-3 text-sm">{task.assignee || '-'}</td>
        <td className="p-3 text-sm">{(task.attempt_number || 0) + 1}</td>
        <td className="p-3 text-sm">{new Date(task.created_at).toLocaleString()}</td>
        <td className="p-3 text-sm">
          {task.output_log_path ? (
            <a
              href={repo.getTaskLogUrl(task.id, 'output')}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-blue-500 no-underline hover:underline"
            >
              View
            </a>
          ) : (
            '-'
          )}
        </td>
      </tr>
      {isExpanded && hasMultipleAttempts && (
        <tr>
          <td colSpan={8} className="p-4 bg-gray-100">
            <h4 className="mb-2">Attempt History</h4>
            {isLoadingAttempts ? (
              <div>Loading attempts...</div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="p-2 text-left">Attempt</th>
                    <th className="p-2 text-left">Status</th>
                    <th className="p-2 text-left">Assignee</th>
                    <th className="p-2 text-left">Assigned</th>
                    <th className="p-2 text-left">Started</th>
                    <th className="p-2 text-left">Finished</th>
                    <th className="p-2 text-left">Logs</th>
                  </tr>
                </thead>
                <tbody>
                  {attempts.map((attempt) => (
                    <tr key={attempt.id}>
                      <td className="p-2">{attempt.attempt_number + 1}</td>
                      <td className="p-2">
                        <TaskBadge task={attempt} size="small" />
                      </td>
                      <td className="p-2">{attempt.assignee || '-'}</td>
                      <td className="p-2">
                        {attempt.assigned_at ? new Date(attempt.assigned_at).toLocaleString() : '-'}
                      </td>
                      <td className="p-2">
                        {attempt.started_at ? new Date(attempt.started_at).toLocaleString() : '-'}
                      </td>
                      <td className="p-2">
                        {attempt.finished_at ? new Date(attempt.finished_at).toLocaleString() : '-'}
                      </td>
                      <td className="p-2">
                        {attempt.output_log_path ? (
                          <a
                            href={repo.getTaskLogUrl(task.id, 'output')}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-500 no-underline hover:underline"
                          >
                            View
                          </a>
                        ) : (
                          '-'
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </td>
        </tr>
      )}
    </Fragment>
  )
}
