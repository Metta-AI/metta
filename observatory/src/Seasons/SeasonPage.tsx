import { FC, useCallback, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

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
  MatchStatus,
  PolicyRow,
  PolicySummary,
  PublicPolicyVersionRow,
  SeasonDetail,
  SeasonMatchSummary,
} from '../repo'
import { formatRelativeTime } from '../utils/datetime'

const StatusBadge: FC<{ status: string }> = ({ status }) => {
  const colors: Record<string, string> = {
    active: 'bg-green-100 text-green-800',
    retired: 'bg-gray-100 text-gray-600',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status] || 'bg-gray-100'}`}>{status}</span>
}

const MatchStatusBadge: FC<{ status: MatchStatus }> = ({ status }) => {
  const colors: Record<MatchStatus, string> = {
    pending: 'bg-gray-100 text-gray-800',
    scheduled: 'bg-blue-100 text-blue-800',
    running: 'bg-yellow-100 text-yellow-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status]}`}>{status}</span>
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
    <div className="space-y-3">
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          <label className="block text-xs text-gray-500 mb-1">Policy</label>
          {selectedPolicy ? (
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{selectedPolicy.name}</span>
              <button
                onClick={() => {
                  setSelectedPolicy(null)
                  setPolicySearch('')
                }}
                className="text-xs text-gray-400 hover:text-gray-600"
              >
                (change)
              </button>
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
          <div className="w-32">
            <label className="block text-xs text-gray-500 mb-1">Version</label>
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
          {submitting ? 'Submitting...' : 'Submit'}
        </Button>
      </div>

      {isAlreadySubmitted && <div className="text-xs text-amber-600">This version is already in the season</div>}
      {submitError && <div className="text-xs text-red-600">{submitError}</div>}
      {submitSuccess && <div className="text-xs text-green-600">{submitSuccess}</div>}
    </div>
  )
}

const formatPolicyDisplay = (p: {
  policy_name: string | null
  policy_version: number | null
  policy_version_id: string
}) => {
  if (p.policy_name && p.policy_version !== null) {
    return `${p.policy_name}:v${p.policy_version}`
  }
  return p.policy_version_id.slice(0, 8)
}

type MatchFilter = {
  pool_name?: string
  policy_version_id?: string
  policy_display?: string
}

export const SeasonPage: FC = () => {
  const { seasonName } = useParams<{ seasonName: string }>()
  const { repo } = useContext(AppContext)

  const [season, setSeason] = useState<SeasonDetail | null>(null)
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [policies, setPolicies] = useState<PolicySummary[]>([])
  const [matches, setMatches] = useState<SeasonMatchSummary[]>([])
  const [matchFilter, setMatchFilter] = useState<MatchFilter>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let ignore = false
    const load = async () => {
      if (!seasonName) return
      try {
        const data = await repo.getSeason(seasonName)
        if (!ignore) {
          setSeason(data)
          setError(null)
        }
      } catch (err: any) {
        if (!ignore) {
          setError(err.message)
        }
      } finally {
        if (!ignore) {
          setLoading(false)
        }
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [repo, seasonName])

  const loadData = useCallback(async () => {
    if (!seasonName) return
    try {
      const [lb, pol] = await Promise.all([repo.getSeasonLeaderboard(seasonName), repo.getSeasonPolicies(seasonName)])
      setLeaderboard(lb)
      setPolicies(pol)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, seasonName, refreshKey])

  const loadMatches = useCallback(async () => {
    if (!seasonName) return
    try {
      const m = await repo.getSeasonMatches(seasonName, {
        pool_name: matchFilter.pool_name,
        policy_version_id: matchFilter.policy_version_id,
        limit: 50,
      })
      setMatches(m)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, seasonName, matchFilter])

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 10000)
    return () => clearInterval(interval)
  }, [loadData])

  useEffect(() => {
    loadMatches()
    const interval = setInterval(loadMatches, 5000)
    return () => clearInterval(interval)
  }, [loadMatches])

  const existingPolicyVersionIds = new Set(policies.map((p) => p.policy_version_id))

  const handleMatchFilterClick = (poolName: string, policyVersionId: string, policyDisplay: string) => {
    setMatchFilter({ pool_name: poolName, policy_version_id: policyVersionId, policy_display: policyDisplay })
  }

  const clearMatchFilter = () => {
    setMatchFilter({})
  }

  if (loading) {
    return (
      <div className="p-6 max-w-5xl mx-auto flex justify-center py-16">
        <Spinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <Card>
          <div className="text-red-600">{error}</div>
        </Card>
      </div>
    )
  }

  if (!season) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <Card>
          <div className="text-gray-500">Season not found</div>
        </Card>
      </div>
    )
  }

  const matchFilterLabel =
    matchFilter.pool_name || matchFilter.policy_version_id
      ? `${matchFilter.pool_name || 'all pools'}${matchFilter.policy_display ? ` / ${matchFilter.policy_display}` : ''}`
      : null

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/seasons" className="text-blue-500 hover:text-blue-700">
          Seasons
        </Link>
        <span className="text-gray-400">/</span>
        <h1 className="text-2xl font-bold text-gray-900">{season.name}</h1>
      </div>

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
                <TR key={entry.policy_version_id}>
                  <TD>{entry.rank}</TD>
                  <TD>
                    <StyledLink to={`/policies/versions/${entry.policy_version_id}`} className="font-mono text-sm">
                      {formatPolicyDisplay(entry)}
                    </StyledLink>
                  </TD>
                  <TD>{entry.score.toPrecision(4)}</TD>
                </TR>
              ))}
            </Table.Body>
          </Table>
        )}
      </Card>

      <Card title="Submit Policy">
        <SubmitForm
          seasonName={seasonName!}
          existingPolicyVersionIds={existingPolicyVersionIds}
          onSubmitted={() => setRefreshKey((k) => k + 1)}
        />
      </Card>

      <Card title="Policies">
        {policies.length === 0 ? (
          <div className="text-gray-500 py-4">No policies submitted yet</div>
        ) : (
          <Table>
            <Table.Header>
              <TH>Policy</TH>
              {season.pools.map((pool) => (
                <TH key={pool} className="capitalize">
                  {pool}
                </TH>
              ))}
            </Table.Header>
            <Table.Body>
              {policies.map((policy) => {
                const poolStatusMap = Object.fromEntries(policy.pools.map((p) => [p.pool_name, p]))
                return (
                  <TR key={policy.policy_version_id}>
                    <TD>
                      <StyledLink to={`/policies/versions/${policy.policy_version_id}`} className="font-mono text-sm">
                        {formatPolicyDisplay(policy)}
                      </StyledLink>
                    </TD>
                    {season.pools.map((poolName) => {
                      const status = poolStatusMap[poolName]
                      if (!status) {
                        return (
                          <TD key={poolName} className="text-gray-400">
                            -
                          </TD>
                        )
                      }
                      return (
                        <TD key={poolName}>
                          <div className="flex flex-col gap-1">
                            <StatusBadge status={status.status} />
                            <button
                              onClick={() =>
                                handleMatchFilterClick(poolName, policy.policy_version_id, formatPolicyDisplay(policy))
                              }
                              className="text-xs text-blue-600 hover:text-blue-800 hover:underline text-left"
                            >
                              {status.matches_completed} matches
                            </button>
                            {status.avg_score !== null && (
                              <span className="text-xs text-gray-500">{status.avg_score.toPrecision(3)} avg</span>
                            )}
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
      </Card>

      <Card
        title={
          <div className="flex items-center gap-2">
            <span>Matches</span>
            {matchFilterLabel && (
              <>
                <span className="text-sm font-normal text-gray-500">({matchFilterLabel})</span>
                <button
                  onClick={clearMatchFilter}
                  className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
                >
                  clear
                </button>
              </>
            )}
          </div>
        }
      >
        {matches.length === 0 ? (
          <div className="text-gray-500 py-4">No matches yet</div>
        ) : (
          <Table>
            <Table.Header>
              <TH>Status</TH>
              <TH>Pool</TH>
              <TH>Players</TH>
              <TH>Scores</TH>
              <TH>Created</TH>
              <TH>Episode</TH>
            </Table.Header>
            <Table.Body>
              {matches.map((match) => (
                <TR key={match.id}>
                  <TD>
                    <MatchStatusBadge status={match.status} />
                  </TD>
                  <TD className="capitalize">{match.pool_name}</TD>
                  <TD>
                    <div className="flex flex-col gap-1">
                      {match.players.map((p, i) => (
                        <StyledLink
                          key={i}
                          to={`/policies/versions/${p.policy_version_id}`}
                          className="font-mono text-xs"
                        >
                          {formatPolicyDisplay(p)}
                        </StyledLink>
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
                  <TD className="text-gray-500 text-sm">{formatRelativeTime(match.created_at)}</TD>
                  <TD>
                    {match.episode_id ? (
                      <StyledLink to={`/episodes/${match.episode_id}`}>View</StyledLink>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </TD>
                </TR>
              ))}
            </Table.Body>
          </Table>
        )}
      </Card>
    </div>
  )
}
