import clsx from 'clsx'
import { FC, Fragment, useCallback, useContext, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { Button } from './components/Button'
import { Card } from './components/Card'
import { Spinner } from './components/Spinner'
import { StyledLink } from './components/StyledLink'
import { Table, TD, TH, TR } from './components/Table'
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

function getTimeDiffColor(from: string | null, to: string | null): string {
  if (!from || !to) return ''
  const fromTs = new Date(from).getTime()
  const toTs = new Date(to).getTime()
  const seconds = Math.floor((toTs - fromTs) / 1000)
  if (seconds < 10) return 'text-green-600'
  if (seconds < 60) return 'text-yellow-600'
  return 'text-red-500'
}

// TODO - consolidate with TaskAttemptTimeline
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

const parsePolicyUri = (uri: string): { name: string; version: string } | null => {
  const match = uri.match(/^metta:\/\/policy\/(.+):v(\d+)$/)
  if (match) {
    return { name: match[1], version: `v${match[2]}` }
  }
  return null
}

const PolicyLink: FC<{ uri: string }> = ({ uri }) => {
  const parsed = parsePolicyUri(uri)
  if (parsed) {
    return (
      <StyledLink to={`/policies?name=${encodeURIComponent(parsed.name)}`} className="font-mono text-xs">
        {parsed.name}:{parsed.version}
      </StyledLink>
    )
  }
  return (
    <span className="font-mono text-xs truncate max-w-[400px]" title={uri}>
      {uri}
    </span>
  )
}

const JobRow: FC<{ job: JobRequest }> = ({ job }) => {
  const [expanded, setExpanded] = useState(false)
  const policyUris = job.job?.policy_uris as string[] | undefined
  const episodeTags = job.job?.episode_tags as Record<string, string> | undefined
  const episodeId = job.result?.episode_id as string | undefined
  const resultError = job.result?.error as string | undefined
  const lifecycleError = job.error

  const hasError = resultError || lifecycleError
  const hasExpandableContent = !episodeId && (hasError || job.result)

  return (
    <Fragment>
      <TR
        className={clsx(hasExpandableContent && 'hover:bg-gray-50 cursor-pointer')}
        onClick={() => hasExpandableContent && setExpanded(!expanded)}
      >
        <TD className="font-mono text-xs">{job.id.slice(0, 8)}</TD>
        <TD>
          <div className="flex flex-col gap-1">
            <StatusBadge status={job.status} />
            {lifecycleError && (
              <span className="text-red-600 text-xs font-medium">{lifecycleError}</span>
            )}
          </div>
        </TD>
        <TD>
          {policyUris?.map((uri, i) => (
            <div key={i}>
              <PolicyLink uri={uri} />
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
            <StyledLink to={`/episodes/${episodeId}`}>View</StyledLink>
          ) : resultError ? (
            <span className="text-red-600 text-xs">{expanded ? '[-] Error' : '[+] Error'}</span>
          ) : job.result ? (
            <span className="text-gray-600 text-xs">{expanded ? '[-] Result' : '[+] Result'}</span>
          ) : (
            '-'
          )}
        </TD>
      </TR>
      {expanded && hasExpandableContent && (
        <tr className="bg-gray-50">
          <td colSpan={6} className="px-3 py-2 space-y-2">
            {lifecycleError && (
              <div>
                <div className="text-xs font-medium text-gray-500 mb-1">Lifecycle Error:</div>
                <pre className="text-xs text-red-600">{lifecycleError}</pre>
              </div>
            )}
            {resultError && (
              <div>
                <div className="text-xs font-medium text-gray-500 mb-1">Result Error:</div>
                <pre className="text-xs whitespace-pre-wrap text-red-600">{resultError}</pre>
              </div>
            )}
            {job.result && !resultError && (
              <div>
                <div className="text-xs font-medium text-gray-500 mb-1">Result:</div>
                <pre className="text-xs whitespace-pre-wrap text-gray-600">
                  {JSON.stringify(job.result, null, 2)}
                </pre>
              </div>
            )}
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
  const [searchParams, setSearchParams] = useSearchParams()
  const [jobs, setJobs] = useState<JobRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<JobStatus | ''>('')
  const [page, setPage] = useState(0)
  const pageSize = 50

  const jobIdFilter = searchParams.get('jobId') || ''

  const handleJobIdChange = (value: string) => {
    if (value) {
      setSearchParams({ jobId: value })
    } else {
      setSearchParams({})
    }
    setPage(0)
  }

  const loadJobs = useCallback(async () => {
    try {
      const statuses = statusFilter ? [statusFilter] : undefined
      const result = await repo.getJobs({
        job_type: 'episode',
        statuses,
        job_id: jobIdFilter || undefined,
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
  }, [repo, statusFilter, jobIdFilter, page])

  useEffect(() => {
    loadJobs()
  }, [loadJobs])

  useEffect(() => {
    const interval = setInterval(loadJobs, 5000)
    return () => clearInterval(interval)
  }, [loadJobs])

  return (
    <div className="p-5 max-w-[1600px] mx-auto">
      <Card title="Episode Jobs">
        {error && (
          <div className="px-4 py-3 mb-5 text-sm bg-red-50 border border-red-400 text-red-800 rounded">{error}</div>
        )}

        <div className="mb-4 flex gap-4 items-center">
          <input
            type="text"
            value={jobIdFilter}
            onChange={(e) => handleJobIdChange(e.target.value)}
            placeholder="Filter by Job ID..."
            className="rounded border h-8 border-gray-300 bg-white text-gray-800 text-sm py-1 px-2 w-80 font-mono"
          />
          <StatusDropdown value={statusFilter} onChange={setStatusFilter} />
          <Button onClick={loadJobs} disabled={loading}>
            Refresh
          </Button>
          {jobIdFilter && (
            <Button onClick={() => handleJobIdChange('')} theme="secondary">
              Clear Filter
            </Button>
          )}
        </div>

        {loading && jobs.length === 0 ? (
          <Spinner size="lg" />
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <Table.Header>
                <TH style={{ width: 100 }}>Job ID</TH>
                <TH style={{ width: 100 }}>Status</TH>
                <TH>Policy URIs</TH>
                <TH style={{ width: 200 }}>Tags</TH>
                <TH style={{ width: 280 }}>Timeline</TH>
                <TH style={{ width: 100 }}>Result</TH>
              </Table.Header>
              <Table.Body>
                {jobs.map((job) => (
                  <JobRow key={job.id} job={job} />
                ))}
              </Table.Body>
            </Table>
            {jobs.length === 0 && <div className="p-5 text-center text-gray-500">No jobs found</div>}
            <div className="flex gap-2 justify-center py-5">
              <Button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
                Previous
              </Button>
              <span className="px-3 py-2 text-sm">Page {page + 1}</span>
              <Button onClick={() => setPage((p) => p + 1)} disabled={jobs.length < pageSize}>
                Next
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
