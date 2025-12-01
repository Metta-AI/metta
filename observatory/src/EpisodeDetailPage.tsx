import { FC, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { ReplayViewer } from './components/ReplayViewer'
import type { EpisodeWithTags, PolicyVersionWithName } from './repo'
import { formatDate, formatRelativeTime } from './utils/datetime'
import { formatPolicyVersion } from './utils/format'

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

export const EpisodeDetailPage: FC = () => {
  const { episodeId } = useParams<{ episodeId: string }>()
  const { repo } = useContext(AppContext)

  const [state, setState] = useState<LoadState<EpisodeWithTags | null>>(() =>
    createInitialState<EpisodeWithTags | null>(null)
  )
  const [policyInfoState, setPolicyInfoState] = useState<LoadState<Record<string, PolicyVersionWithName | null>>>(() =>
    createInitialState<Record<string, PolicyVersionWithName | null>>({})
  )

  useEffect(() => {
    if (!episodeId) {
      setState({ data: null, loading: false, error: 'Missing episode id' })
      return
    }

    let isMounted = true

    const loadEpisode = async () => {
      setState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const response = await repo.queryEpisodes({ episode_ids: [episodeId], limit: 1, offset: 0 })
        const episode = response.episodes[0] ?? null
        if (!isMounted) return
        if (!episode) {
          setState({ data: null, loading: false, error: 'Episode not found' })
          return
        }
        setState({ data: episode, loading: false, error: null })
      } catch (error: any) {
        if (isMounted) {
          setState({
            data: null,
            loading: false,
            error: error.message ?? 'Failed to load episode',
          })
        }
      }
    }

    void loadEpisode()
    return () => {
      isMounted = false
    }
  }, [episodeId, repo])

  useEffect(() => {
    const episode = state.data
    if (!episode) {
      setPolicyInfoState((prev) => ({ ...prev, loading: false, data: {} }))
      return
    }

    const policyIds = Object.keys(episode.avg_rewards || {})
    if (policyIds.length === 0) {
      setPolicyInfoState((prev) => ({ ...prev, loading: false, data: {} }))
      return
    }

    let isMounted = true
    const loadPolicyInfo = async () => {
      setPolicyInfoState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const entries = await Promise.all(
          policyIds.map(async (policyId) => {
            try {
              const info = await repo.getPolicyVersion(policyId)
              return [policyId, info] as const
            } catch (error: any) {
              console.error('Failed to load policy version', policyId, error)
              return [policyId, null] as const
            }
          })
        )
        if (isMounted) {
          setPolicyInfoState({ data: Object.fromEntries(entries), loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setPolicyInfoState({
            data: {},
            loading: false,
            error: error.message ?? 'Failed to load policy details',
          })
        }
      }
    }

    void loadPolicyInfo()
    return () => {
      isMounted = false
    }
  }, [repo, state.data])

  const episode = state.data

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Episode</p>
          <h1 className="text-2xl font-semibold text-gray-900 break-all">{episodeId ?? 'Episode Detail'}</h1>
          {episode && (
            <div className="flex flex-wrap gap-3 text-sm text-gray-600">
              <span title={formatDate(episode.created_at)}>Created: {formatRelativeTime(episode.created_at)}</span>
              {episode.eval_task_id && <span>Eval Task: {episode.eval_task_id}</span>}
              <span className="flex items-center gap-1 text-gray-500">
                Episode ID:
                <span className="font-mono text-xs text-gray-700">{episode.id}</span>
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          {episode?.primary_pv_id ? (
            <Link
              to={`/policies/versions/${episode.primary_pv_id}`}
              className="inline-flex items-center px-3 py-2 rounded border border-blue-500 text-blue-600 no-underline hover:bg-blue-50 text-sm"
            >
              View Primary Policy
            </Link>
          ) : null}
          <Link
            to="/leaderboard"
            className="inline-flex items-center px-3 py-2 rounded border border-gray-300 text-gray-700 no-underline hover:bg-gray-50 text-sm"
          >
            Back to Leaderboard
          </Link>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Policies & Scores</h2>
          </div>
        </div>
        <div className="p-5">
          {state.loading ? (
            <div className="text-gray-500 text-sm">Loading episode...</div>
          ) : state.error ? (
            <div className="text-red-600 text-sm">{state.error}</div>
          ) : !episode ? (
            <div className="text-gray-500 text-sm">Episode not found.</div>
          ) : Object.keys(episode.avg_rewards || {}).length === 0 ? (
            <div className="text-gray-500 text-sm">No policy metrics recorded for this episode.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-50 text-left text-xs font-semibold uppercase text-gray-600">
                    <th className="px-3 py-2 border-b border-gray-200">Policy</th>
                    <th className="px-3 py-2 border-b border-gray-200">Policy ID</th>
                    <th className="px-3 py-2 border-b border-gray-200">Avg Reward</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(episode.avg_rewards || {})
                    .sort(([, a], [, b]) => (typeof b === 'number' && typeof a === 'number' ? b - a : 0))
                    .map(([policyId, reward]) => {
                      const info = policyInfoState.data[policyId]
                      const policyLabel = formatPolicyVersion(info)
                      const isPrimary = episode.primary_pv_id === policyId
                      return (
                        <tr key={policyId} className="border-b border-gray-100 align-top">
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <Link
                                to={`/policies/versions/${policyId}`}
                                className="text-blue-600 no-underline hover:underline font-medium"
                              >
                                {policyLabel}
                              </Link>
                              {isPrimary ? (
                                <span className="text-[10px] uppercase tracking-wide text-blue-600 border border-blue-200 rounded px-1 py-0.5">
                                  Primary
                                </span>
                              ) : null}
                              {policyInfoState.loading && !info ? (
                                <span className="text-xs text-gray-400">(loading details)</span>
                              ) : null}
                            </div>
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-gray-700">{policyId}</td>
                          <td className="px-3 py-2 font-mono">{formatScore(reward)}</td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
              {policyInfoState.error ? (
                <div className="text-red-600 text-xs mt-2">Failed to load policy details: {policyInfoState.error}</div>
              ) : null}
            </div>
          )}
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Replay</h2>
          </div>
        </div>
        <div className="p-5 space-y-3">
          {state.loading ? (
            <div className="text-gray-500 text-sm">Loading replay...</div>
          ) : state.error ? (
            <div className="text-red-600 text-sm">{state.error}</div>
          ) : !episode || !episode.replay_url ? (
            <div className="text-gray-500 text-sm">No replay available for this episode.</div>
          ) : (
            <ReplayViewer replayUrl={episode.replay_url} label="Episode replay" />
          )}
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="p-5 space-y-4">
          {state.loading ? (
            <div className="text-gray-500 text-sm">Loading episode...</div>
          ) : state.error ? (
            <div className="text-red-600 text-sm">{state.error}</div>
          ) : episode ? (
            <>
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Tags</h3>
                {Object.keys(episode.tags).length === 0 ? (
                  <div className="text-gray-500 text-sm">No tags found.</div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(episode.tags)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([key, value]) => (
                        <span
                          key={`${episode.id}-${key}-${value}`}
                          className="inline-flex items-center px-2 py-1 text-xs rounded bg-gray-100 border border-gray-200"
                        >
                          {key}: {value}
                        </span>
                      ))}
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Attributes</h3>
                {Object.keys(episode.attributes || {}).length === 0 ? (
                  <div className="text-gray-500 text-sm">No attributes recorded.</div>
                ) : (
                  <pre className="bg-gray-50 border border-gray-200 rounded p-3 text-xs overflow-auto">
                    {JSON.stringify(episode.attributes, null, 2)}
                  </pre>
                )}
              </div>
            </>
          ) : (
            <div className="text-gray-500 text-sm">Episode not found.</div>
          )}
        </div>
      </div>
    </div>
  )
}
