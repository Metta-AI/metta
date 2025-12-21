import clsx from 'clsx'
import { FC, Fragment, useState } from 'react'
import { Link } from 'react-router-dom'

import { TaskBadge } from '../components/TaskBadge'
import { EvalTask, PublicPolicyVersionRow, Repo, TaskAttempt } from '../repo'
import { formatDate, formatDurationBetween } from '../utils/datetime'
import { parsePolicyVersionId } from './TasksTable'

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

type TaskRowProps = {
  task: EvalTask
  repo: Repo
  policyInfoMap: Record<string, PublicPolicyVersionRow>
  attemptedPolicyIds: Set<string>
}

const getTimeDiffColor = (from: string | null, to: string | null): string => {
  if (!from || !to) return ''
  const fromTs = new Date(from).getTime()
  const toTs = new Date(to).getTime()
  const minutes = Math.floor((toTs - fromTs) / 1000 / 60)
  if (minutes < 2) return 'text-green-600'
  if (minutes < 10) return 'text-yellow-600'
  return 'text-red-500'
}

const parseRecipe = (command: string): string | null => {
  const match = command.match(/run\.py\s+(\S+)/)
  if (match) {
    return match[1].replace(/^recipes\.(experiment|prod)\./, '')
  }
  return null
}

export const TaskRow: FC<TaskRowProps> = ({ task, repo, policyInfoMap, attemptedPolicyIds }) => {
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
  const policyVersionId = parsePolicyVersionId(task.command)
  const policyInfo = policyVersionId ? policyInfoMap[policyVersionId] : null
  const hasAttemptedPolicy = policyVersionId ? attemptedPolicyIds.has(policyVersionId) : true

  return (
    <Fragment>
      <tr
        className={clsx('border-b border-gray-200', 'cursor-pointer', 'hover:bg-gray-50')}
        onClick={() => toggleTaskExpansion()}
      >
        <td className="p-3 text-sm">
          <span className="mr-2 text-xs text-gray-300 group-hover:text-gray-400">{isExpanded ? '▾' : '▸'}</span>
          {policyInfo ? (
            <Link
              to={`/policies/versions/${policyVersionId}`}
              onClick={(e) => e.stopPropagation()}
              className="text-blue-500 no-underline hover:underline"
            >
              {policyInfo.name}:v{policyInfo.version}
            </Link>
          ) : policyVersionId && !hasAttemptedPolicy ? (
            <span className="text-gray-400 text-xs">Loading...</span>
          ) : (
            '-'
          )}
          {isLoadingAttempts && <span className="ml-2 text-xs text-gray-500">...</span>}
        </td>
        <td className="p-3 text-sm truncate" title={task.command}>
          {parseRecipe(task.command) || '-'}
        </td>
        <td className="p-3">
          <TaskBadge task={task} />
        </td>
        <td className="p-3 text-sm">{formatDate(task.created_at)}</td>
        <td className="p-3 text-sm">{formatDurationBetween(task.created_at, task.finished_at) || '-'}</td>
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
          <td colSpan={6} className="p-4 bg-gray-100">
            <div className="mb-2 text-xs text-gray-500">Task ID: {task.id}</div>
            <div className="mb-3 text-xs text-gray-700 font-mono break-all">{task.command}</div>
            {isLoadingAttempts ? (
              <div>Loading attempts...</div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="p-2 text-left">Attempt</th>
                    <th className="p-2 text-left">Status</th>
                    <th className="p-2 text-left">Assignee</th>
                    <th className="p-2 text-left">Timeline</th>
                    <th className="p-2 text-left">Logs</th>
                  </tr>
                </thead>
                <tbody>
                  {attempts.map((attempt) => {
                    const ddUrl = getDatadogLogsUrl(attempt)
                    const assignedDiff = formatDurationBetween(task.created_at, attempt.assigned_at)
                    const startedDiff = formatDurationBetween(attempt.assigned_at, attempt.started_at)
                    const finishedDiff = formatDurationBetween(attempt.started_at, attempt.finished_at)
                    return (
                      <tr key={attempt.id} className="align-top">
                        <td className="p-2">{attempt.attempt_number + 1}</td>
                        <td className="p-2">
                          <TaskBadge task={attempt} size="small" />
                        </td>
                        <td className="p-2">{attempt.assignee || '-'}</td>
                        <td className="p-2">
                          <table className="text-xs">
                            <tbody>
                              <tr>
                                <td className="pr-2 text-gray-600">Assigned:</td>
                                <td className="text-right">{formatDate(attempt.assigned_at)}</td>
                                <td
                                  className={clsx(
                                    'pl-2 text-right',
                                    getTimeDiffColor(task.created_at, attempt.assigned_at)
                                  )}
                                >
                                  {assignedDiff ? `(+ ${assignedDiff})` : ''}
                                </td>
                              </tr>
                              <tr>
                                <td className="pr-2 text-gray-600">Started:</td>
                                <td className="text-right">{formatDate(attempt.started_at)}</td>
                                <td
                                  className={clsx(
                                    'pl-2 text-right',
                                    getTimeDiffColor(attempt.assigned_at, attempt.started_at)
                                  )}
                                >
                                  {startedDiff ? `(+ ${startedDiff})` : ''}
                                </td>
                              </tr>
                              <tr>
                                <td className="pr-2 text-gray-600">Finished:</td>
                                <td className="text-right">{formatDate(attempt.finished_at)}</td>
                                <td
                                  className={clsx(
                                    'pl-2 text-right',
                                    getTimeDiffColor(attempt.started_at, attempt.finished_at)
                                  )}
                                >
                                  {finishedDiff ? `(+ ${finishedDiff})` : ''}
                                </td>
                              </tr>
                            </tbody>
                          </table>
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
