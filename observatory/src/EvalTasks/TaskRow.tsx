import clsx from 'clsx'
import { FC, Fragment, useState } from 'react'

import { TaskBadge } from '../components/TaskBadge'
import { EvalTask, Repo, TaskAttempt } from '../repo'

type DatadogLogsParams = {
  assignee: string | null
  assigned_at: string | null
  finished_at: string | null
}

const getDatadogLogsUrl = (params: DatadogLogsParams): string | null => {
  if (!params.assignee || !params.assigned_at) {
    return null
  }

  const assignedAt = new Date(params.assigned_at).getTime()
  const fromTs = assignedAt - 60 * 60 * 1000
  const toTs = params.finished_at
    ? new Date(params.finished_at).getTime() + 60 * 60 * 1000
    : assignedAt + 3 * 60 * 60 * 1000

  const urlParams = new URLSearchParams({
    query: `pod_name:${params.assignee}`,
    agg_m: 'count',
    agg_m_source: 'base',
    agg_t: 'count',
    cols: 'host,service',
    fromUser: 'true',
    messageDisplay: 'inline',
    refresh_mode: 'sliding',
    storage: 'hot',
    stream_sort: 'time_ascending',
    viz: 'stream',
    from_ts: fromTs.toString(),
    to_ts: toTs.toString(),
    live: 'true',
  })

  return `https://app.datadoghq.com/logs?${urlParams.toString()}`
}

export const TaskRow: FC<{ task: EvalTask; repo: Repo }> = ({ task, repo }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  // UI state
  const [attempts, setAttempts] = useState<TaskAttempt[]>([])
  const [isLoadingAttempts, setIsLoadingAttempts] = useState(false)

  const toggleTaskExpansion = async () => {
    if (isExpanded) {
      setIsExpanded(false)
    } else {
      if (attempts.length === 0 && !isLoadingAttempts) {
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

  const taskDatadogUrl = getDatadogLogsUrl(task)

  return (
    <Fragment>
      <tr className={clsx('border-b border-gray-200', 'cursor-pointer')} onClick={() => toggleTaskExpansion()}>
        <td className="p-3">
          <span className="mr-2 text-xs text-gray-500">{isExpanded ? '▼' : '▶'}</span>
          {task.id}
          {isLoadingAttempts && <span className="ml-2 text-xs text-gray-500">...</span>}
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
          <div className="flex gap-1 items-center">
            {taskDatadogUrl ? (
              <a
                href={taskDatadogUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-blue-500 no-underline hover:underline"
              >
                Datadog
              </a>
            ) : null}
            {taskDatadogUrl && task.output_log_path ? <span className="text-gray-400">|</span> : null}
            {task.output_log_path ? (
              <a
                href={repo.getTaskLogUrl(task.id, 'output')}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-blue-500 no-underline hover:underline"
              >
                Output
              </a>
            ) : null}
            {!task.output_log_path && !taskDatadogUrl ? '-' : null}
          </div>
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={8} className="p-4 bg-gray-100">
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
                  {attempts.map((attempt) => {
                    const ddUrl = getDatadogLogsUrl(attempt)
                    return (
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
                          <div className="flex gap-1 items-center">
                            {ddUrl ? (
                              <a
                                href={ddUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-500 no-underline hover:underline"
                              >
                                Datadog
                              </a>
                            ) : null}
                            {ddUrl && attempt.output_log_path ? <span className="text-gray-400">|</span> : null}
                            {attempt.output_log_path ? (
                              <a
                                href={repo.getTaskLogUrl(task.id, 'output')}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-500 no-underline hover:underline"
                              >
                                Output
                              </a>
                            ) : null}
                            {!attempt.output_log_path && !ddUrl ? '-' : null}
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </td>
        </tr>
      )}
    </Fragment>
  )
}
