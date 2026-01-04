import { FC, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
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
  pool_names: string[]
  policy_version_ids: string[]
}

export const SeasonPage: FC = () => {
  const { seasonName } = useParams<{ seasonName: string }>()
  const { repo } = useContext(AppContext)

  const [season, setSeason] = useState<SeasonDetail | null>(null)
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [policies, setPolicies] = useState<PolicySummary[]>([])
  const [allMatches, setAllMatches] = useState<SeasonMatchSummary[]>([])
  const [matchFilter, setMatchFilter] = useState<MatchFilter>({ pool_names: [], policy_version_ids: [] })
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
      const m = await repo.getSeasonMatches(seasonName, { limit: 100 })
      setAllMatches(m)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, seasonName])

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

  const filteredMatches = useMemo(() => {
    return allMatches.filter((m) => {
      if (matchFilter.pool_names.length > 0 && !matchFilter.pool_names.includes(m.pool_name)) {
        return false
      }
      if (matchFilter.policy_version_ids.length > 0) {
        const matchPvIds = m.players.map((p) => p.policy_version_id)
        if (!matchFilter.policy_version_ids.some((id) => matchPvIds.includes(id))) {
          return false
        }
      }
      return true
    })
  }, [allMatches, matchFilter])

  const matchCountsByPolicyPool = useMemo(() => {
    const counts: Record<string, { scored: number; pending: number }> = {}
    for (const m of allMatches) {
      for (const p of m.players) {
        const key = `${p.policy_version_id}:${m.pool_name}`
        if (!counts[key]) counts[key] = { scored: 0, pending: 0 }
        if (m.status === 'completed' && p.score !== null) {
          counts[key].scored++
        } else if (m.status !== 'failed') {
          counts[key].pending++
        }
      }
    }
    return counts
  }, [allMatches])

  const leaderboardScoreByPolicy = useMemo(() => {
    const map: Record<string, number> = {}
    for (const entry of leaderboard) {
      map[entry.policy_version_id] = entry.score
    }
    return map
  }, [leaderboard])

  const handleMatchFilterClick = (poolName: string, policyVersionId: string) => {
    setMatchFilter({ pool_names: [poolName], policy_version_ids: [policyVersionId] })
  }

  const poolOptions = (season?.pools || []).map((p) => ({ value: p, label: p }))
  const playerOptions = policies.map((p) => ({
    value: p.policy_version_id,
    label: formatPolicyDisplay(p),
  }))

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
                  <TD>
                    {entry.score.toPrecision(4)}{' '}
                    <span
                      onClick={() => handleMatchFilterClick('competition', entry.policy_version_id)}
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
                      const pool = poolStatusMap[poolName]
                      if (!pool) {
                        return (
                          <TD key={poolName} className="text-gray-400">
                            -
                          </TD>
                        )
                      }
                      const counts = matchCountsByPolicyPool[`${policy.policy_version_id}:${poolName}`] || {
                        scored: 0,
                        pending: 0,
                      }
                      const avgScore =
                        poolName === 'competition' ? leaderboardScoreByPolicy[policy.policy_version_id] : undefined
                      return (
                        <TD key={poolName}>
                          <div className="flex flex-col gap-1.5 items-start">
                            <Link to={`/seasons/${seasonName}/players/${policy.policy_version_id}`}>
                              <span
                                className={`inline-block px-2 py-1 rounded text-xs font-medium transition-colors ${
                                  pool.active
                                    ? 'bg-green-100 text-green-800 hover:bg-green-200'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                              >
                                {pool.active ? 'active' : 'retired'}
                              </span>
                            </Link>
                            <div className="text-sm">
                              {avgScore !== undefined ? avgScore.toPrecision(4) : '-'}{' '}
                              <span
                                onClick={() => handleMatchFilterClick(poolName, policy.policy_version_id)}
                                className="text-gray-400 hover:text-blue-600 cursor-pointer transition-colors"
                              >
                                ({counts.scored} scored{counts.pending > 0 && `, ${counts.pending} pending`})
                              </span>
                            </div>
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
          seasonName={seasonName!}
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
                    </TD>
                    <TD className="capitalize">{match.pool_name}</TD>
                    <TD className="text-right">
                      <div className="flex flex-col gap-1 items-end">
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
      </Card>
    </div>
  )
}
