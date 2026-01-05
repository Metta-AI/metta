import { FC, useCallback, useContext, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import Select from 'react-select'

import { AppContext } from '../AppContext'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { Input } from '../components/Input'
import { Spinner } from '../components/Spinner'
import { StyledLink } from '../components/StyledLink'
import { Table, TH, TR, TD } from '../components/Table'
import { useDebouncedValue } from '../hooks/useDebouncedValue'
import {
  LeaderboardEntry,
  PolicyRow,
  PolicySummary,
  PublicPolicyVersionRow,
  SeasonDetail,
  SeasonMatchSummary,
} from '../repo'
import { formatRelativeTime } from '../utils/datetime'

const MatchStatusBadge: FC<{ status: string }> = ({ status }) => {
  const colors: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-800',
    scheduled: 'bg-blue-100 text-blue-800',
    running: 'bg-yellow-100 text-yellow-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status] || 'bg-gray-100'}`}>{status}</span>
}

const selectStyles = {
  control: (base: any) => ({
    ...base,
    minHeight: '32px',
    fontSize: '0.75rem',
  }),
  valueContainer: (base: any) => ({
    ...base,
    padding: '0 6px',
  }),
  multiValue: (base: any) => ({
    ...base,
    backgroundColor: '#dbeafe',
  }),
  multiValueLabel: (base: any) => ({
    ...base,
    color: '#1e40af',
    fontSize: '0.75rem',
    padding: '1px 4px',
  }),
  multiValueRemove: (base: any) => ({
    ...base,
    color: '#1e40af',
    ':hover': {
      backgroundColor: '#bfdbfe',
      color: '#1e3a8a',
    },
  }),
  option: (base: any) => ({
    ...base,
    fontSize: '0.75rem',
    padding: '6px 10px',
  }),
  placeholder: (base: any) => ({
    ...base,
    fontSize: '0.75rem',
  }),
}

const seasonSelectStyles = {
  control: (base: any) => ({
    ...base,
    minHeight: '40px',
    fontSize: '1rem',
    fontWeight: 600,
    minWidth: '200px',
  }),
  singleValue: (base: any) => ({
    ...base,
    fontWeight: 600,
  }),
  option: (base: any) => ({
    ...base,
    fontSize: '1rem',
    padding: '8px 12px',
  }),
}

const SubmitForm: FC<{
  seasonName: string
  existingPolicyVersionIds: Set<string>
  onSubmitted: () => void
}> = ({ seasonName, existingPolicyVersionIds, onSubmitted }) => {
  const { repo } = useContext(AppContext)
  const [policySearch, setPolicySearch] = useState('')
  const [policies, setPolicies] = useState<PolicyRow[]>([])
  const [selectedPolicy, setSelectedPolicy] = useState<PolicyRow | null>(null)
  const [versions, setVersions] = useState<PublicPolicyVersionRow[]>([])
  const [selectedVersion, setSelectedVersion] = useState<PublicPolicyVersionRow | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitSuccess, setSubmitSuccess] = useState<string | null>(null)

  const debouncedSearch = useDebouncedValue(policySearch, 300)

  useEffect(() => {
    if (!debouncedSearch) {
      setPolicies([])
      return
    }
    let ignore = false
    repo.getPolicies({ name_fuzzy: debouncedSearch, limit: 10 }).then((res) => {
      if (!ignore) setPolicies(res.entries)
    })
    return () => {
      ignore = true
    }
  }, [repo, debouncedSearch])

  useEffect(() => {
    if (!selectedPolicy) {
      setVersions([])
      setSelectedVersion(null)
      return
    }
    let ignore = false
    repo.getVersionsForPolicy(selectedPolicy.id, { limit: 50 }).then((res) => {
      if (!ignore) {
        setVersions(res.entries)
        if (res.entries.length > 0) {
          setSelectedVersion(res.entries[0])
        }
      }
    })
    return () => {
      ignore = true
    }
  }, [repo, selectedPolicy])

  useEffect(() => {
    if (!submitSuccess && !submitError) return
    const timer = setTimeout(() => {
      setSubmitSuccess(null)
      setSubmitError(null)
    }, 10000)
    return () => clearTimeout(timer)
  }, [submitSuccess, submitError])

  const handleSubmit = async () => {
    if (!selectedVersion) return
    setSubmitting(true)
    setSubmitError(null)
    setSubmitSuccess(null)
    try {
      const result = await repo.submitToSeason(seasonName, selectedVersion.id)
      setSubmitSuccess(`Submitted to pools: ${result.pools.join(', ')}`)
      setSelectedPolicy(null)
      setSelectedVersion(null)
      setPolicySearch('')
      onSubmitted()
    } catch (err: any) {
      setSubmitError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const isAlreadySubmitted = selectedVersion && existingPolicyVersionIds.has(selectedVersion.id)

  return (
    <div className="border-t border-gray-200 pt-4 mt-4">
      <div className="text-xs text-gray-500 mb-2">Submit new player</div>
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          {selectedPolicy ? (
            <div className="flex items-center gap-1">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-100 text-blue-800 rounded text-xs">
                {selectedPolicy.name}
                <button
                  onClick={() => {
                    setSelectedPolicy(null)
                    setPolicySearch('')
                  }}
                  className="hover:text-blue-600 font-medium"
                >
                  x
                </button>
              </span>
            </div>
          ) : (
            <div className="relative">
              <Input value={policySearch} onChange={setPolicySearch} placeholder="Search policies..." size="sm" />
              {policies.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg max-h-48 overflow-auto">
                  {policies.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setSelectedPolicy(p)
                        setPolicies([])
                      }}
                      className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 border-b border-gray-100 last:border-0"
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {selectedPolicy && versions.length > 0 && (
          <div className="w-24">
            <select
              value={selectedVersion?.id || ''}
              onChange={(e) => {
                const v = versions.find((v) => v.id === e.target.value)
                setSelectedVersion(v || null)
              }}
              className="w-full text-sm px-2 py-1 border border-gray-300 rounded-sm focus:outline-none focus:border-blue-500"
            >
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  v{v.version}
                </option>
              ))}
            </select>
          </div>
        )}

        <Button
          onClick={handleSubmit}
          theme="primary"
          size="sm"
          disabled={!selectedVersion || submitting || !!isAlreadySubmitted}
        >
          {submitting ? '...' : 'Submit'}
        </Button>
      </div>

      {isAlreadySubmitted && <div className="text-xs text-amber-600 mt-1">Already in season</div>}
      {submitError && <div className="text-xs text-red-600 mt-1">{submitError}</div>}
      {submitSuccess && <div className="text-xs text-green-600 mt-1">{submitSuccess}</div>}
    </div>
  )
}

const formatPolicyDisplay = (p: { policy: { id: string; name: string | null; version: number | null } }) => {
  if (p.policy.name && p.policy.version !== null) {
    return `${p.policy.name}:v${p.policy.version}`
  }
  return p.policy.id.slice(0, 8)
}

type MatchFilter = {
  pool_names: string[]
  policy_version_ids: string[]
}

export const SeasonsPage: FC = () => {
  const { seasonName: urlSeasonName } = useParams<{ seasonName?: string }>()
  const navigate = useNavigate()
  const { repo } = useContext(AppContext)

  const [seasons, setSeasons] = useState<SeasonDetail[]>([])
  const [selectedSeasonName, setSelectedSeasonName] = useState<string | null>(urlSeasonName || null)
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [policies, setPolicies] = useState<PolicySummary[]>([])
  const [filteredMatches, setFilteredMatches] = useState<SeasonMatchSummary[]>([])
  const [matchPage, setMatchPage] = useState(0)
  const [hasMoreMatches, setHasMoreMatches] = useState(true)
  const [matchFilter, setMatchFilter] = useState<MatchFilter>({ pool_names: [], policy_version_ids: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let ignore = false
    repo.getSeasons().then((data) => {
      if (!ignore) {
        setSeasons(data)
        setLoading(false)
      }
    })
    return () => {
      ignore = true
    }
  }, [repo])

  useEffect(() => {
    if (urlSeasonName) {
      setSelectedSeasonName(urlSeasonName)
    } else if (seasons.length > 0 && !selectedSeasonName) {
      setSelectedSeasonName(seasons[0].name)
    }
  }, [urlSeasonName, seasons, selectedSeasonName])

  useEffect(() => {
    setMatchPage(0)
  }, [selectedSeasonName, matchFilter])

  const season = seasons.find((s) => s.name === selectedSeasonName) || null

  const loadData = useCallback(async () => {
    if (!selectedSeasonName) return
    try {
      const [lb, pol] = await Promise.all([
        repo.getSeasonLeaderboard(selectedSeasonName),
        repo.getSeasonPolicies(selectedSeasonName),
      ])
      setLeaderboard(lb)
      setPolicies(pol)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, selectedSeasonName, refreshKey])

  const MATCHES_PAGE_SIZE = 50

  const loadFilteredMatches = useCallback(async () => {
    if (!selectedSeasonName) return
    try {
      const m = await repo.getSeasonMatches(selectedSeasonName, {
        limit: MATCHES_PAGE_SIZE,
        offset: matchPage * MATCHES_PAGE_SIZE,
        pool_names: matchFilter.pool_names.length > 0 ? matchFilter.pool_names : undefined,
        policy_version_ids: matchFilter.policy_version_ids.length > 0 ? matchFilter.policy_version_ids : undefined,
      })
      setFilteredMatches(m)
      setHasMoreMatches(m.length === MATCHES_PAGE_SIZE)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, selectedSeasonName, matchPage, matchFilter])

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 10000)
    return () => clearInterval(interval)
  }, [loadData])

  useEffect(() => {
    loadFilteredMatches()
    const interval = setInterval(loadFilteredMatches, 5000)
    return () => clearInterval(interval)
  }, [loadFilteredMatches])

  const existingPolicyVersionIds = new Set(policies.map((p) => p.policy.id))

  const handleMatchFilterClick = (poolName: string, policyVersionId: string) => {
    setMatchFilter({ pool_names: [poolName], policy_version_ids: [policyVersionId] })
  }

  const handleSeasonChange = (option: { value: string; label: string } | null) => {
    if (option) {
      setSelectedSeasonName(option.value)
      navigate(`/tournament/${option.value}`)
    }
  }

  const seasonOptions = seasons.map((s) => ({ value: s.name, label: s.name }))
  const poolNames = season?.pools || []
  const poolOptions = poolNames.map((p) => ({ value: p, label: p }))
  const playerOptions = policies.map((p) => ({
    value: p.policy.id,
    label: formatPolicyDisplay(p),
  }))

  if (seasons.length === 0 && loading) {
    return (
      <div className="p-6 max-w-5xl mx-auto flex justify-center py-16">
        <Spinner size="lg" />
      </div>
    )
  }

  if (error && !season) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <Card>
          <div className="text-red-600">{error}</div>
        </Card>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <span className="text-gray-600 font-medium">Season:</span>
          <Select
            options={seasonOptions}
            value={seasonOptions.find((o) => o.value === selectedSeasonName) || null}
            onChange={handleSeasonChange}
            styles={seasonSelectStyles}
            isSearchable={false}
            placeholder="Select season..."
          />
        </div>
        {season?.description && (
          <div className="text-gray-500 text-sm">
            {season.description.summary && <div>{season.description.summary}</div>}
            {season.description.pools.length > 0 && (
              <div className="mt-1 ml-4 space-y-0.5">
                {season.description.pools.map((pool) => (
                  <div key={pool.name}>
                    <span className="font-medium text-gray-600">{pool.name}:</span> {pool.description}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {!season ? (
        <Card>
          <div className="text-gray-500 py-4">Select a season to view</div>
        </Card>
      ) : (
        <>
          <Card title="Leaderboard">
            {leaderboard.length === 0 ? (
              <div className="text-gray-500 py-4">No entries yet</div>
            ) : (
              <Table>
                <Table.Header>
                  <TH>Rank</TH>
                  <TH>Policy</TH>
                  <TH>Score</TH>
                </Table.Header>
                <Table.Body>
                  {leaderboard.map((entry) => (
                    <TR key={entry.policy.id}>
                      <TD>{entry.rank}</TD>
                      <TD>
                        <StyledLink to={`/policies/versions/${entry.policy.id}`} className="font-mono text-sm">
                          {formatPolicyDisplay(entry)}
                        </StyledLink>
                      </TD>
                      <TD>
                        {entry.score.toPrecision(4)}{' '}
                        <span
                          onClick={() => handleMatchFilterClick('competition', entry.policy.id)}
                          className="text-gray-400 hover:text-blue-600 cursor-pointer transition-colors"
                        >
                          ({entry.matches} matches)
                        </span>
                      </TD>
                    </TR>
                  ))}
                </Table.Body>
              </Table>
            )}
          </Card>

          <Card title="Players">
            {policies.length === 0 ? (
              <div className="text-gray-500 py-4">No players submitted yet</div>
            ) : (
              <Table>
                <Table.Header>
                  <TH>Player</TH>
                  <TH>Entered</TH>
                  {poolNames.map((poolName) => (
                    <TH key={poolName} className="capitalize">
                      {poolName}
                    </TH>
                  ))}
                </Table.Header>
                <Table.Body>
                  {policies.map((policy) => {
                    const poolStatusMap = Object.fromEntries(policy.pools.map((p) => [p.pool_name, p]))
                    return (
                      <TR key={policy.policy.id}>
                        <TD>
                          <StyledLink to={`/policies/versions/${policy.policy.id}`} className="font-mono text-sm">
                            {formatPolicyDisplay(policy)}
                          </StyledLink>
                        </TD>
                        <TD className="text-gray-500 text-sm">{formatRelativeTime(policy.entered_at)}</TD>
                        {poolNames.map((poolName) => {
                          const pool = poolStatusMap[poolName]
                          if (!pool) {
                            return (
                              <TD key={poolName} className="text-gray-400">
                                -
                              </TD>
                            )
                          }
                          return (
                            <TD key={poolName}>
                              <div className="flex flex-col gap-1.5 items-start">
                                <span
                                  className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                                    pool.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                                  }`}
                                >
                                  {pool.active ? 'active' : 'retired'}
                                </span>
                                <span
                                  onClick={() => handleMatchFilterClick(poolName, policy.policy.id)}
                                  className="text-sm text-gray-400 hover:text-blue-600 cursor-pointer transition-colors"
                                >
                                  ({pool.completed} matches
                                  {pool.failed > 0 && `, ${pool.failed} failed`}
                                  {pool.pending > 0 && `, ${pool.pending} pending`})
                                </span>
                              </div>
                            </TD>
                          )
                        })}
                      </TR>
                    )
                  })}
                </Table.Body>
              </Table>
            )}
            <SubmitForm
              seasonName={selectedSeasonName!}
              existingPolicyVersionIds={existingPolicyVersionIds}
              onSubmitted={() => setRefreshKey((k) => k + 1)}
            />
          </Card>

          <Card title="Matches">
            <div className="flex gap-4 mb-4 pb-4 border-b border-gray-100">
              <div className="flex-1">
                <div className="text-xs text-gray-500 mb-1">Pool</div>
                <Select
                  isMulti
                  options={poolOptions}
                  value={poolOptions.filter((o) => matchFilter.pool_names.includes(o.value))}
                  onChange={(selected) => setMatchFilter((f) => ({ ...f, pool_names: selected.map((s) => s.value) }))}
                  placeholder="All pools"
                  styles={selectStyles}
                  isClearable
                />
              </div>
              <div className="flex-1">
                <div className="text-xs text-gray-500 mb-1">Players</div>
                <Select
                  isMulti
                  options={playerOptions}
                  value={playerOptions.filter((o) => matchFilter.policy_version_ids.includes(o.value))}
                  onChange={(selected) =>
                    setMatchFilter((f) => ({ ...f, policy_version_ids: selected.map((s) => s.value) }))
                  }
                  placeholder="All players"
                  styles={selectStyles}
                  isClearable
                />
              </div>
            </div>
            {filteredMatches.length === 0 ? (
              <div className="text-gray-500 py-4">No matches</div>
            ) : (
              <Table>
                <Table.Header>
                  <TH className="w-24">Created</TH>
                  <TH className="w-20">Status</TH>
                  <TH className="w-24">Pool</TH>
                  <TH className="text-right">Players</TH>
                  <TH className="w-16">Agents</TH>
                  <TH className="w-20">Score</TH>
                </Table.Header>
                <Table.Body>
                  {filteredMatches.map((match) => {
                    const agentCounts = match.players.map(
                      (p) => match.assignments.filter((a) => a === p.policy_index).length
                    )
                    return (
                      <TR key={match.id}>
                        <TD className="text-gray-500 text-sm">{formatRelativeTime(match.created_at)}</TD>
                        <TD>
                          <div className="flex justify-between gap-1">
                            {match.status === 'completed' && match.episode_id ? (
                              <Link
                                to={`/episodes/${match.episode_id}`}
                                className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-800 hover:bg-green-200 transition-colors"
                              >
                                Results
                              </Link>
                            ) : (
                              <MatchStatusBadge status={match.status} />
                            )}
                            {match.job_id && (
                              <Link
                                to={`/episode-jobs?jobId=${match.job_id}`}
                                className="px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
                              >
                                Job
                              </Link>
                            )}
                          </div>
                        </TD>
                        <TD>
                          <span
                            onClick={() => setMatchFilter((f) => ({ ...f, pool_names: [match.pool_name] }))}
                            className="cursor-pointer hover:text-blue-600 transition-colors"
                          >
                            {match.pool_name}
                          </span>
                        </TD>
                        <TD className="text-right">
                          <div className="flex flex-col gap-1 items-end">
                            {match.players.map((p, i) => (
                              <span
                                key={i}
                                onClick={() => setMatchFilter((f) => ({ ...f, policy_version_ids: [p.policy.id] }))}
                                className="font-mono text-xs cursor-pointer hover:text-blue-600 transition-colors"
                              >
                                {formatPolicyDisplay(p)}
                              </span>
                            ))}
                          </div>
                        </TD>
                        <TD>
                          <div className="flex flex-col gap-1 font-mono text-xs">
                            {match.players.map((_, i) => (
                              <span key={i}>{agentCounts[i]}</span>
                            ))}
                          </div>
                        </TD>
                        <TD>
                          <div className="flex flex-col gap-1 font-mono text-xs">
                            {match.players.map((p, i) => (
                              <span key={i}>{p.score !== null ? p.score.toPrecision(3) : '-'}</span>
                            ))}
                          </div>
                        </TD>
                      </TR>
                    )
                  })}
                </Table.Body>
              </Table>
            )}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-100">
              <Button size="sm" onClick={() => setMatchPage((p) => Math.max(0, p - 1))} disabled={matchPage === 0}>
                Previous
              </Button>
              <span className="text-sm text-gray-500">Page {matchPage + 1}</span>
              <Button size="sm" onClick={() => setMatchPage((p) => p + 1)} disabled={!hasMoreMatches}>
                Next
              </Button>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
