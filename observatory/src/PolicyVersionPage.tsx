import { FC, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { CopyableUri } from './components/CopyableUri'
import { ReplayViewer, normalizeReplayUrl } from './components/ReplayViewer'
import { TaskBadge } from './components/TaskBadge'
import { LEADERBOARD_SIM_NAME_EPISODE_KEY } from './constants'
import type { EpisodeWithTags, EvalTask, LeaderboardPolicyEntry, PolicyVersionWithName } from './repo'
import { formatDate, formatRelativeTime } from './utils/datetime'
import { formatPolicyVersion } from './utils/format'

const TASK_PAGE_SIZE = 100

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
  const [taskState, setTaskState] = useState<LoadState<EvalTask[]>>(() => createInitialState<EvalTask[]>([]))
  const [episodesState, setEpisodesState] = useState<LoadState<EpisodeWithTags[]>>(() =>
    createInitialState<EpisodeWithTags[]>([])
  )
  const [episodeReplayPreview, setEpisodeReplayPreview] = useState<{ url: string; label: string } | null>(null)

  useEffect(() => {
    if (!policyVersionId) {
      setPolicyState({ data: null, loading: false, error: 'Missing policy version id' })
      setTaskState({ data: [], loading: false, error: 'Missing policy version id' })
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

    const loadTasks = async () => {
      setTaskState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const tasks: EvalTask[] = []
        let page = 1

        while (true) {
          const response = await repo.getEvalTasksPaginated(page, TASK_PAGE_SIZE, { command: policyVersionId })
          tasks.push(...response.tasks)

          if (page >= response.total_pages) {
            break
          }
          page += 1
        }

        if (isMounted) {
          setTaskState({ data: tasks, loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setTaskState({
            data: [],
            loading: false,
            error: error.message ?? 'Failed to load related tasks',
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

    void loadPolicy()
    void loadTasks()
    void loadEpisodes()
    void loadPolicyVersionInfo()

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

  const getLeaderboardTagDisplay = (episode: EpisodeWithTags): { value: string; tooltip: string } => {
    const tags = episode.tags || {}
    const leaderboardValue = tags[LEADERBOARD_SIM_NAME_EPISODE_KEY] ?? '—'
    const otherTags = Object.entries(tags).filter(([key]) => key !== LEADERBOARD_SIM_NAME_EPISODE_KEY)
    const tooltip =
      otherTags.length === 0 ? 'No other tags' : otherTags.map(([key, value]) => `${key}: ${value}`).join('\n')
    return { value: leaderboardValue, tooltip }
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
        <div className="flex items-center gap-3">
          {pvInfo && (
            <Link
              to={`/policies/${pvInfo.policy_id}`}
              className="inline-flex items-center px-3 py-2 rounded border border-gray-300 text-gray-700 no-underline hover:bg-gray-50 text-sm"
            >
              ← Back to policy
            </Link>
          )}
        </div>
      </div>

      {pvInfo && <CopyableUri uri={`metta://policy/${pvInfo.name}:v${pvInfo.version}`} />}

      {policyState.error ? (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded text-sm">
          {policyState.error}
        </div>
      ) : null}

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Episodes</h2>
          </div>
        </div>
        <div className="p-5">
          {episodesState.loading ? (
            <div className="text-gray-500 text-sm">Loading episodes...</div>
          ) : episodesState.error ? (
            <div className="text-red-600 text-sm">{episodesState.error}</div>
          ) : episodesState.data.length === 0 ? (
            <div className="text-gray-500 text-sm">No episodes found for this policy version.</div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="bg-gray-50 text-left text-xs font-semibold uppercase text-gray-600">
                      <th className="px-3 py-2 border-b border-gray-200">ID</th>
                      <th className="px-3 py-2 border-b border-gray-200">Leaderboard Tag</th>
                      <th className="px-3 py-2 border-b border-gray-200">Replay</th>
                      <th className="px-3 py-2 border-b border-gray-200">Created</th>
                      <th className="px-3 py-2 border-b border-gray-200">Avg Reward (policy)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {episodesState.data.map((episode) => (
                      <tr key={episode.id} className="border-b border-gray-100 align-top">
                        <td className="px-3 py-2">
                          <Link
                            to={`/episodes/${episode.id}`}
                            className="font-mono text-xs text-blue-600 no-underline hover:underline break-words max-w-xs"
                          >
                            {episode.id}
                          </Link>
                        </td>
                        <td className="px-3 py-2">
                          {(() => {
                            const { value, tooltip } = getLeaderboardTagDisplay(episode)
                            return (
                              <span
                                className="inline-flex items-center px-2 py-1 text-xs rounded bg-gray-100 border border-gray-200 font-mono"
                                title={tooltip}
                              >
                                {value}
                              </span>
                            )
                          })()}
                        </td>
                        <td className="px-3 py-2">
                          {(() => {
                            const replayUrl = normalizeReplayUrl(episode.replay_url)
                            if (!replayUrl) {
                              return '—'
                            }
                            return (
                              <div className="flex items-center gap-2">
                                <a
                                  href={replayUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-600 no-underline hover:underline"
                                >
                                  Replay
                                </a>
                                <button
                                  type="button"
                                  className="px-2 py-1 text-xs rounded border border-blue-200 text-blue-700 bg-white hover:bg-blue-50"
                                  onClick={() => toggleEpisodeReplayPreview(episode)}
                                >
                                  Show below
                                </button>
                              </div>
                            )
                          })()}
                        </td>
                        <td className="px-3 py-2" title={formatDate(episode.created_at)}>
                          {formatRelativeTime(episode.created_at)}
                        </td>
                        <td className="px-3 py-2 font-mono">{formatScore(getPolicyAvgReward(episode))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Related Jobs</h2>
          </div>
        </div>
        <div className="p-5">
          {taskState.loading ? (
            <div className="text-gray-500 text-sm">Loading tasks...</div>
          ) : taskState.error ? (
            <div className="text-red-600 text-sm">{taskState.error}</div>
          ) : taskState.data.length === 0 ? (
            <div className="text-gray-500 text-sm">No tasks found for this policy version.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-50 text-left text-xs font-semibold uppercase text-gray-600">
                    <th className="px-3 py-2 border-b border-gray-200">ID</th>
                    <th className="px-3 py-2 border-b border-gray-200">Status</th>
                    <th className="px-3 py-2 border-b border-gray-200">Assignee</th>
                    <th className="px-3 py-2 border-b border-gray-200">Attempts</th>
                    <th className="px-3 py-2 border-b border-gray-200">Created</th>
                    <th className="px-3 py-2 border-b border-gray-200">Logs</th>
                  </tr>
                </thead>
                <tbody>
                  {taskState.data.map((task) => (
                    <tr key={task.id} className="border-b border-gray-100 align-top">
                      <td className="px-3 py-2">
                        <span className="cursor-default" title={task.command}>
                          {task.id}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <TaskBadge task={task} size="small" />
                      </td>
                      <td className="px-3 py-2">{task.assignee || '—'}</td>
                      <td className="px-3 py-2">{(task.attempt_number || 0) + 1}</td>
                      <td className="px-3 py-2" title={formatDate(task.created_at)}>
                        {formatRelativeTime(task.created_at)}
                      </td>
                      <td className="px-3 py-2">
                        {task.output_log_path ? (
                          <a
                            href={repo.getTaskLogUrl(task.id, 'output')}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 no-underline hover:underline"
                          >
                            View
                          </a>
                        ) : (
                          '—'
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
