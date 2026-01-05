import { FC, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { A } from './components/A'
import { Button } from './components/Button'
import { Card } from './components/Card'
import { CopyableUri } from './components/CopyableUri'
import { LinkButton } from './components/LinkButton'
import { normalizeReplayUrl, ReplayViewer } from './components/ReplayViewer'
import { Spinner } from './components/Spinner'
import { StyledLink } from './components/StyledLink'
import { Table, TD, TH, TR } from './components/Table'
import { TasksTable } from './EvalTasks/TasksTable'
import type { EpisodeWithTags, LeaderboardPolicyEntry, MembershipHistoryEntry, PolicyVersionWithName } from './repo'
import { formatDate, formatRelativeTime } from './utils/datetime'
import { formatPolicyVersion } from './utils/format'

const ActionBadge: FC<{ action: string }> = ({ action }) => {
  const colors: Record<string, string> = {
    add: 'bg-green-100 text-green-800',
    remove: 'bg-gray-100 text-gray-600',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[action] || 'bg-gray-100'}`}>{action}</span>
}

type LoadState<T> = {
  data: T
  loading: boolean
  error: string | null
}

const createInitialState = <T,>(data: T): LoadState<T> => ({
  data,
  loading: true,
  error: null,
})

const formatScore = (value: number | null | undefined): string => {
  if (typeof value !== 'number') {
    return '—'
  }
  return value.toFixed(2)
}

export const PolicyVersionPage: FC = () => {
  const { policyVersionId } = useParams<{ policyVersionId: string }>()
  const { repo } = useContext(AppContext)

  const [policyState, setPolicyState] = useState<LoadState<LeaderboardPolicyEntry | null>>(() =>
    createInitialState<LeaderboardPolicyEntry | null>(null)
  )
  const [policyVersionInfo, setPolicyVersionInfo] = useState<LoadState<PolicyVersionWithName | null>>(() =>
    createInitialState<PolicyVersionWithName | null>(null)
  )
  const [taskError, setTaskError] = useState<string | null>(null)
  const [episodesState, setEpisodesState] = useState<LoadState<EpisodeWithTags[]>>(() =>
    createInitialState<EpisodeWithTags[]>([])
  )
  const [membershipsState, setMembershipsState] = useState<LoadState<MembershipHistoryEntry[]>>(() =>
    createInitialState<MembershipHistoryEntry[]>([])
  )
  const [episodeReplayPreview, setEpisodeReplayPreview] = useState<{ url: string; label: string } | null>(null)

  useEffect(() => {
    if (!policyVersionId) {
      setPolicyState({ data: null, loading: false, error: 'Missing policy version id' })
      return
    }

    let isMounted = true

    const loadPolicy = async () => {
      setPolicyState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const response = await repo.getLeaderboardPolicy(policyVersionId)
        const matchingEntry =
          response.entries.find((entry) => entry.policy_version.id === policyVersionId) ?? response.entries[0]

        if (!isMounted) {
          return
        }

        if (!matchingEntry) {
          setPolicyState({ data: null, loading: false, error: null })
          return
        }

        setPolicyState({ data: matchingEntry, loading: false, error: null })
      } catch (error: any) {
        if (isMounted) {
          setPolicyState({
            data: null,
            loading: false,
            error: error.message ?? 'Failed to load policy details',
          })
        }
      }
    }

    const loadEpisodes = async () => {
      setEpisodesState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const response = await repo.queryEpisodes({
          primary_policy_version_ids: [policyVersionId],
          limit: 200,
          offset: 0,
        })
        if (isMounted) {
          setEpisodesState({ data: response.episodes, loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setEpisodesState({
            data: [],
            loading: false,
            error: error.message ?? 'Failed to load episodes',
          })
        }
      }
    }

    const loadPolicyVersionInfo = async () => {
      setPolicyVersionInfo((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const info = await repo.getPolicyVersion(policyVersionId)
        if (isMounted) {
          setPolicyVersionInfo({ data: info, loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setPolicyVersionInfo({
            data: null,
            loading: false,
            error: error.message ?? 'Failed to load policy version info',
          })
        }
      }
    }

    const loadMemberships = async () => {
      setMembershipsState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const memberships = await repo.getPlayerMemberships(policyVersionId)
        if (isMounted) {
          setMembershipsState({ data: memberships, loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setMembershipsState({
            data: [],
            loading: false,
            error: error.message ?? 'Failed to load memberships',
          })
        }
      }
    }

    void loadPolicy()
    void loadEpisodes()
    void loadPolicyVersionInfo()
    void loadMemberships()

    return () => {
      isMounted = false
    }
  }, [policyVersionId, repo])

  const policyEntry = policyState.data
  const leaderboardPolicyVersion = policyEntry?.policy_version
  const pvInfo = policyVersionInfo.data
  const policyCreatedAt = leaderboardPolicyVersion?.policy_created_at || pvInfo?.created_at || null
  const policyDisplay = policyVersionInfo.loading ? 'Loading...' : formatPolicyVersion(pvInfo, policyVersionId)

  useEffect(() => {
    document.title = policyDisplay ? `${policyDisplay} | Observatory` : 'Policy Version | Observatory'
  }, [policyDisplay])

  const getPolicyAvgReward = (episode: EpisodeWithTags): number | undefined => {
    if (policyVersionId && episode.avg_rewards[policyVersionId] !== undefined) {
      return episode.avg_rewards[policyVersionId]
    }
    if (episode.primary_pv_id && episode.avg_rewards[episode.primary_pv_id] !== undefined) {
      return episode.avg_rewards[episode.primary_pv_id]
    }
    return undefined
  }

  const toggleEpisodeReplayPreview = (episode: EpisodeWithTags) => {
    const normalized = normalizeReplayUrl(episode.replay_url)
    if (!normalized) {
      return
    }
    setEpisodeReplayPreview((prev) => {
      if (prev?.url === normalized) {
        return null
      }
      return { url: normalized, label: `Episode ${episode.id.slice(0, 8)}` }
    })
  }

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Policy Version</p>
          <h1 className="text-2xl font-semibold text-gray-900">{policyDisplay}</h1>
          {!policyVersionInfo.loading && (
            <div className="flex flex-wrap gap-3 text-sm text-gray-600">
              {policyCreatedAt && <span className="text-gray-500">Created: {formatDate(policyCreatedAt)}</span>}
              {pvInfo && (
                <span className="flex items-center gap-1 text-gray-500">
                  Policy Version ID:
                  <span className="font-mono text-xs text-gray-700">{pvInfo.id}</span>
                </span>
              )}
            </div>
          )}
        </div>
        {pvInfo && (
          <LinkButton to={`/policies/${pvInfo.policy_id}`} theme="tertiary">
            ← Back to policy
          </LinkButton>
        )}
      </div>

      {pvInfo && <CopyableUri uri={`metta://policy/${pvInfo.name}:v${pvInfo.version}`} />}

      {policyState.error ? (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded text-sm">
          {policyState.error}
        </div>
      ) : null}

      <Card title="Episodes">
        {episodesState.loading ? (
          <Spinner />
        ) : episodesState.error ? (
          <div className="text-red-600 text-sm">{episodesState.error}</div>
        ) : episodesState.data.length === 0 ? (
          <div className="text-gray-500 text-sm">No episodes found for this policy version.</div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <Table>
                <Table.Header>
                  <TH>ID</TH>
                  <TH>Replay</TH>
                  <TH>Created</TH>
                  <TH>Avg Reward (policy)</TH>
                </Table.Header>
                <Table.Body>
                  {episodesState.data.map((episode) => (
                    <TR key={episode.id}>
                      <TD>
                        <StyledLink to={`/episodes/${episode.id}`} className="font-mono text-xs">
                          {episode.id}
                        </StyledLink>
                      </TD>
                      <TD>
                        {(() => {
                          const replayUrl = normalizeReplayUrl(episode.replay_url)
                          if (!replayUrl) {
                            return '—'
                          }
                          return (
                            <div className="flex items-center gap-2">
                              <A href={replayUrl} target="_blank" rel="noopener noreferrer">
                                Replay
                              </A>
                              <Button size="sm" onClick={() => toggleEpisodeReplayPreview(episode)}>
                                Show below
                              </Button>
                            </div>
                          )
                        })()}
                      </TD>
                      <TD title={formatDate(episode.created_at)}>{formatRelativeTime(episode.created_at)}</TD>
                      <TD className="px-3 py-2">
                        <span className="font-mono">{formatScore(getPolicyAvgReward(episode))}</span>
                      </TD>
                    </TR>
                  ))}
                </Table.Body>
              </Table>
            </div>
            {episodeReplayPreview ? (
              <div className="mt-4">
                <ReplayViewer
                  replayUrl={episodeReplayPreview.url}
                  label={`Replay preview (${episodeReplayPreview.label})`}
                />
              </div>
            ) : null}
          </>
        )}
      </Card>

      <Card title="Tournament Memberships">
        {membershipsState.loading ? (
          <Spinner />
        ) : membershipsState.error ? (
          <div className="text-red-600 text-sm">{membershipsState.error}</div>
        ) : membershipsState.data.length === 0 ? (
          <div className="text-gray-500 text-sm">No tournament memberships found for this policy version.</div>
        ) : (
          <Table>
            <Table.Header>
              <TH>Time</TH>
              <TH>Season</TH>
              <TH>Pool</TH>
              <TH>Action</TH>
              <TH>Notes</TH>
            </Table.Header>
            <Table.Body>
              {membershipsState.data.map((entry, i) => (
                <TR key={i}>
                  <TD className="text-gray-500 text-sm">{formatRelativeTime(entry.created_at)}</TD>
                  <TD>
                    <Link to={`/tournament/${entry.season_name}`} className="text-blue-500 hover:underline">
                      {entry.season_name}
                    </Link>
                  </TD>
                  <TD className="capitalize">{entry.pool_name}</TD>
                  <TD>
                    <ActionBadge action={entry.action} />
                  </TD>
                  <TD className="text-sm text-gray-600">{entry.notes || '-'}</TD>
                </TR>
              ))}
            </Table.Body>
          </Table>
        )}
      </Card>

      {taskError && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded text-sm">{taskError}</div>
      )}

      {policyVersionId && (
        <Card title="Tasks">
          <TasksTable repo={repo} setError={setTaskError} initialFilters={{ command: policyVersionId }} hideFilters />
        </Card>
      )}
    </div>
  )
}
