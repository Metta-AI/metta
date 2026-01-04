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
  MatchPlayerSummary,
  MatchStatus,
  MatchSummary,
  PolicyRow,
  PoolMember,
  PublicPolicyVersionRow,
  SeasonDetail,
} from '../repo'
import { formatRelativeTime } from '../utils/datetime'

const StatusBadge: FC<{ status: MatchStatus }> = ({ status }) => {
  const colors: Record<MatchStatus, string> = {
    pending: 'bg-gray-100 text-gray-800',
    scheduled: 'bg-blue-100 text-blue-800',
    running: 'bg-yellow-100 text-yellow-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status]}`}>{status}</span>
}

const formatPlayerDisplay = (p: MatchPlayerSummary): string => {
  if (p.policy_name && p.policy_version !== null) {
    return `${p.policy_name}:v${p.policy_version}`
  }
  return p.policy_version_id.slice(0, 8)
}

const PlayerLink: FC<{ player: MatchPlayerSummary }> = ({ player }) => (
  <StyledLink to={`/policies/versions/${player.policy_version_id}`} className="font-mono text-xs">
    {formatPlayerDisplay(player)}
  </StyledLink>
)

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
      setSubmitSuccess(`Submitted to pools: ${result.pool_names.join(', ')}`)
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

export const SeasonPage: FC = () => {
  const { seasonName } = useParams<{ seasonName: string }>()
  const { repo } = useContext(AppContext)

  const [season, setSeason] = useState<SeasonDetail | null>(null)
  const [selectedPool, setSelectedPool] = useState<string>('')
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [matches, setMatches] = useState<MatchSummary[]>([])
  const [members, setMembers] = useState<PoolMember[]>([])
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
          if (data.pools.length > 0) {
            setSelectedPool(data.pools[0])
          }
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

  const loadLeaderboard = useCallback(async () => {
    if (!seasonName) return
    try {
      const lb = await repo.getSeasonLeaderboard(seasonName)
      setLeaderboard(lb.entries)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, seasonName, refreshKey])

  const loadPoolData = useCallback(async () => {
    if (!seasonName || !selectedPool) return
    try {
      const [m, mem] = await Promise.all([
        repo.getPoolMatches(seasonName, selectedPool, { limit: 20 }),
        repo.getPoolMembers(seasonName, selectedPool),
      ])
      setMatches(m)
      setMembers(mem)
    } catch (err: any) {
      setError(err.message)
    }
  }, [repo, seasonName, selectedPool, refreshKey])

  useEffect(() => {
    loadLeaderboard()
    const interval = setInterval(loadLeaderboard, 10000)
    return () => clearInterval(interval)
  }, [loadLeaderboard])

  useEffect(() => {
    loadPoolData()
    const interval = setInterval(loadPoolData, 10000)
    return () => clearInterval(interval)
  }, [loadPoolData])

  const existingPolicyVersionIds = new Set(leaderboard.map((e) => e.policy_version_id))

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
              {leaderboard.map((entry) => {
                const displayName =
                  entry.policy_name && entry.policy_version !== null
                    ? `${entry.policy_name}:v${entry.policy_version}`
                    : entry.policy_version_id.slice(0, 8)
                return (
                  <TR key={entry.policy_version_id}>
                    <TD>{entry.rank}</TD>
                    <TD>
                      <StyledLink to={`/policies/versions/${entry.policy_version_id}`} className="font-mono text-sm">
                        {displayName}
                      </StyledLink>
                    </TD>
                    <TD>{entry.score.toPrecision(4)}</TD>
                  </TR>
                )
              })}
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

      <hr className="border-gray-200" />

      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-600 mr-1">Pool:</span>
        {season.pools.map((pool) => (
          <button
            key={pool}
            onClick={() => setSelectedPool(pool)}
            className={`px-3 py-1 text-sm rounded-full border transition-colors capitalize ${
              selectedPool === pool
                ? 'border-blue-600 text-blue-600 bg-blue-50'
                : 'border-gray-300 text-gray-600 bg-white hover:border-gray-400'
            }`}
          >
            {pool}
          </button>
        ))}
      </div>

      <div className="space-y-6">
        <Card title="Match History">
          {matches.length === 0 ? (
            <div className="text-gray-500 py-4">No matches yet</div>
          ) : (
            <Table>
              <Table.Header>
                <TH>Status</TH>
                <TH>Policy</TH>
                <TH>Agents</TH>
                <TH>Score</TH>
                <TH>Tags</TH>
                <TH>Created</TH>
                <TH>Episode</TH>
              </Table.Header>
              <Table.Body>
                {matches.map((match) => {
                  const tags = match.episode_tags || {}
                  const displayTags = Object.entries(tags).filter(([k]) => k !== 'match_id' && k !== 'pool_id')
                  const agentCounts = match.assignments.reduce(
                    (acc, policyIdx) => {
                      acc[policyIdx] = (acc[policyIdx] || 0) + 1
                      return acc
                    },
                    {} as Record<number, number>
                  )
                  return (
                    <TR key={match.id}>
                      <TD>
                        <StatusBadge status={match.status} />
                      </TD>
                      <TD>
                        <div className="flex flex-col gap-1">
                          {match.players.map((p, i) => (
                            <PlayerLink key={i} player={p} />
                          ))}
                        </div>
                      </TD>
                      <TD>
                        <div className="flex flex-col gap-1 text-xs text-gray-600">
                          {match.players.map((p, i) => (
                            <span key={i}>{agentCounts[p.policy_index] || 0}</span>
                          ))}
                        </div>
                      </TD>
                      <TD>
                        <div className="flex flex-col gap-1 font-mono text-xs">
                          {match.players.map((p, i) => (
                            <span key={i}>{p.score !== null ? p.score.toPrecision(4) : '-'}</span>
                          ))}
                        </div>
                      </TD>
                      <TD>
                        {displayTags.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {displayTags.map(([k, v]) => (
                              <span
                                key={k}
                                className="px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded"
                                title={`${k}: ${v}`}
                              >
                                {k}={v}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
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
                  )
                })}
              </Table.Body>
            </Table>
          )}
        </Card>

        <Card title="Pool Members">
          {members.length === 0 ? (
            <div className="text-gray-500 py-4">No members yet</div>
          ) : (
            <Table>
              <Table.Header>
                <TH>Policy</TH>
                <TH>Added</TH>
                <TH>Status</TH>
              </Table.Header>
              <Table.Body>
                {members.map((member) => {
                  const displayName =
                    member.policy_name && member.policy_version !== null
                      ? `${member.policy_name}:v${member.policy_version}`
                      : member.policy_version_id.slice(0, 8)
                  return (
                    <TR key={member.policy_version_id}>
                      <TD>
                        <StyledLink to={`/policies/versions/${member.policy_version_id}`} className="font-mono text-sm">
                          {displayName}
                        </StyledLink>
                      </TD>
                      <TD className="text-gray-500 text-sm">{formatRelativeTime(member.added_at)}</TD>
                      <TD>
                        {member.retired ? (
                          <span className="text-xs text-gray-500">
                            Retired {member.retired_at ? formatRelativeTime(member.retired_at) : ''}
                          </span>
                        ) : (
                          <span className="text-xs text-green-600">Active</span>
                        )}
                      </TD>
                    </TR>
                  )
                })}
              </Table.Body>
            </Table>
          )}
        </Card>
      </div>
    </div>
  )
}
