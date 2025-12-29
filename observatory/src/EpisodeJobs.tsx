import clsx from 'clsx'
import { FC, PropsWithChildren, useCallback, useContext, useEffect, useState, Fragment } from 'react'
import { Link } from 'react-router-dom'

import { AppContext } from './AppContext'
import { JobRequest, JobStatus } from './repo'
import { formatDate, formatDurationBetween } from './utils/datetime'

const StatusBadge: FC<{ status: JobStatus }> = ({ status }) => {
  const colors: Record<JobStatus, string> = {
    pending: 'bg-gray-100 text-gray-800',
    dispatched: 'bg-blue-100 text-blue-800',
    running: 'bg-yellow-100 text-yellow-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status]}`}>{status}</span>
}

const TH: FC<PropsWithChildren<{ width?: string | number }>> = ({ children, width }) => {
  return (
    <th
      className="px-3 pt-2 pb-2 text-left text-xs text-gray-800 font-semibold tracking-wide uppercase"
      style={{ width }}
    >
      {children}
    </th>
  )
}

const TD: FC<PropsWithChildren<{ className?: string }>> = ({ children, className = '' }) => {
  return <td className={`px-3 py-2 text-sm ${className}`}>{children}</td>
}

const getTimeDiffColor = (from: string | null, to: string | null): string => {
  if (!from || !to) return ''
  const fromTs = new Date(from).getTime()
  const toTs = new Date(to).getTime()
  const seconds = Math.floor((toTs - fromTs) / 1000)
  if (seconds < 10) return 'text-green-600'
  if (seconds < 60) return 'text-yellow-600'
  return 'text-red-500'
}

const SimpleButton: FC<{
  onClick: () => void
  disabled?: boolean
  children: React.ReactNode
}> = ({ onClick, disabled, children }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-3 py-1 text-sm rounded border border-gray-300 bg-white ${
        disabled ? 'opacity-50 cursor-not-allowed text-gray-400' : 'hover:bg-gray-50 text-gray-700 cursor-pointer'
      }`}
    >
      {children}
    </button>
  )
}

const Timeline: FC<{ job: JobRequest }> = ({ job }) => {
  const dispatchedDiff = formatDurationBetween(job.created_at, job.dispatched_at)
  const runningDiff = formatDurationBetween(job.dispatched_at, job.running_at)
  const completedDiff = formatDurationBetween(job.running_at, job.completed_at)

  return (
    <table className="text-xs">
      <tbody>
        <tr>
          <td className="pr-2 text-gray-600">Created:</td>
          <td className="text-right">{formatDate(job.created_at)}</td>
          <td className="pl-2 w-16"></td>
        </tr>
        <tr>
          <td className="pr-2 text-gray-600">Dispatched:</td>
          <td className="text-right">{formatDate(job.dispatched_at)}</td>
          <td className={clsx('pl-2 text-right', getTimeDiffColor(job.created_at, job.dispatched_at))}>
            {dispatchedDiff ? `+${dispatchedDiff}` : ''}
          </td>
        </tr>
        <tr>
          <td className="pr-2 text-gray-600">Running:</td>
          <td className="text-right">{formatDate(job.running_at)}</td>
          <td className={clsx('pl-2 text-right', getTimeDiffColor(job.dispatched_at, job.running_at))}>
            {runningDiff ? `+${runningDiff}` : ''}
          </td>
        </tr>
        <tr>
          <td className="pr-2 text-gray-600">Completed:</td>
          <td className="text-right">{formatDate(job.completed_at)}</td>
          <td className={clsx('pl-2 text-right', getTimeDiffColor(job.running_at, job.completed_at))}>
            {completedDiff ? `+${completedDiff}` : ''}
          </td>
        </tr>
      </tbody>
    </table>
  )
}

const JobRow: FC<{ job: JobRequest }> = ({ job }) => {
  const [expanded, setExpanded] = useState(false)
  const policyUris = job.job?.policy_uris as string[] | undefined
  const episodeTags = job.job?.episode_tags as Record<string, string> | undefined
  const episodeId = job.result?.episode_id as string | undefined
  const error = job.result?.error as string | undefined

  const hasExpandableContent = !episodeId && (error || job.result)

  return (
    <Fragment>
      <tr
        className={clsx(
          'border-b border-gray-200 hover:bg-gray-50 align-top',
          hasExpandableContent && 'cursor-pointer'
        )}
        onClick={() => hasExpandableContent && setExpanded(!expanded)}
      >
        <TD className="font-mono text-xs">{job.id.slice(0, 8)}</TD>
        <TD>
          <StatusBadge status={job.status} />
        </TD>
        <TD>
          {policyUris?.map((uri, i) => (
            <div key={i} className="font-mono text-xs truncate max-w-[400px]" title={uri}>
              {uri}
            </div>
          ))}
        </TD>
        <TD>
          {episodeTags && Object.keys(episodeTags).length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {Object.entries(episodeTags).map(([k, v]) => (
                <span key={k} className="px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded" title={`${k}: ${v}`}>
                  {k}={v}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-gray-400">-</span>
          )}
        </TD>
        <TD>
          <Timeline job={job} />
        </TD>
        <TD>
          {episodeId ? (
            <Link to={`/episodes/${episodeId}`} className="text-blue-600 hover:underline">
              View
            </Link>
          ) : error ? (
            <span className="text-red-600 text-xs">
              {expanded ? '[-] Error' : '[+] Error'}
            </span>
          ) : job.result ? (
            <span className="text-gray-600 text-xs">
              {expanded ? '[-] Result' : '[+] Result'}
            </span>
          ) : (
            '-'
          )}
        </TD>
      </tr>
      {expanded && hasExpandableContent && (
        <tr className="bg-gray-50">
          <td colSpan={6} className="px-3 py-2">
            <pre className={clsx('text-xs whitespace-pre-wrap', error ? 'text-red-600' : 'text-gray-600')}>
              {error || JSON.stringify(job.result, null, 2)}
            </pre>
          </td>
        </tr>
      )}
    </Fragment>
  )
}

const StatusDropdown: FC<{ value: JobStatus | ''; onChange: (value: JobStatus | '') => void }> = ({
  value,
  onChange,
}) => {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as JobStatus | '')}
      className="rounded border h-8 border-gray-300 bg-white text-gray-800 text-sm py-1 px-2"
    >
      <option value="">All Statuses</option>
      <option value="pending">Pending</option>
      <option value="dispatched">Dispatched</option>
      <option value="running">Running</option>
      <option value="completed">Completed</option>
      <option value="failed">Failed</option>
    </select>
  )
}

export const EpisodeJobs: FC = () => {
  const { repo } = useContext(AppContext)
  const [jobs, setJobs] = useState<JobRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<JobStatus | ''>('')
  const [page, setPage] = useState(0)
  const pageSize = 50

  const loadJobs = useCallback(async () => {
    try {
      const statuses = statusFilter ? [statusFilter] : undefined
      const result = await repo.getJobs({
        job_type: 'episode',
        statuses,
        limit: pageSize,
        offset: page * pageSize,
      })
      setJobs(result)
      setError(null)
    } catch (err: any) {
      setError(`Failed to load jobs: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [repo, statusFilter, page])

  useEffect(() => {
    loadJobs()
  }, [loadJobs])

  useEffect(() => {
    const interval = setInterval(loadJobs, 5000)
    return () => clearInterval(interval)
  }, [loadJobs])

  return (
    <div className="p-5 max-w-[1600px] mx-auto">
      <h1 className="mb-5">Episode Jobs</h1>

      {error && (
        <div className="px-4 py-3 mb-5 text-sm bg-red-50 border border-red-400 text-red-800 rounded">{error}</div>
      )}

      <div className="mb-4 flex gap-4 items-center">
        <StatusDropdown value={statusFilter} onChange={setStatusFilter} />
        <SimpleButton onClick={loadJobs} disabled={loading}>
          Refresh
        </SimpleButton>
      </div>

      {loading && jobs.length === 0 ? (
        <div>Loading jobs...</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead className="border-b border-b-gray-300 bg-gray-100">
              <tr>
                <TH width={100}>Job ID</TH>
                <TH width={100}>Status</TH>
                <TH>Policy URIs</TH>
                <TH width={200}>Tags</TH>
                <TH width={280}>Timeline</TH>
                <TH width={100}>Result</TH>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <JobRow key={job.id} job={job} />
              ))}
            </tbody>
          </table>
          {jobs.length === 0 && <div className="p-5 text-center text-gray-500">No jobs found</div>}
        </div>
      )}

      <div className="flex gap-2 justify-center py-5">
        <SimpleButton onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
          Previous
        </SimpleButton>
        <span className="px-3 py-2 text-sm">Page {page + 1}</span>
        <SimpleButton onClick={() => setPage((p) => p + 1)} disabled={jobs.length < pageSize}>
          Next
        </SimpleButton>
      </div>
    </div>
  )
}
