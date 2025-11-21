import { FC, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import type { EpisodeWithTags } from './repo'

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

const formatDate = (value: string | null): string => {
  if (!value) {
    return '—'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString()
}

const formatRelativeTime = (value: string | null): string => {
  if (!value) {
    return '—'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  const diffMs = Date.now() - date.getTime()
  const diffSeconds = Math.max(0, Math.floor(diffMs / 1000))

  if (diffSeconds < 60) return 'just now'
  const diffMinutes = Math.floor(diffSeconds / 60)
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return `${diffDays}d ago`
  const diffWeeks = Math.floor(diffDays / 7)
  if (diffWeeks < 4) return `${diffWeeks}w ago`
  const diffMonths = Math.floor(diffDays / 30)
  if (diffMonths < 12) return `${diffMonths}mo ago`
  const diffYears = Math.floor(diffDays / 365)
  return `${diffYears}y ago`
}

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

  const episode = state.data

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Episode</p>
          <h1 className="text-2xl font-semibold text-gray-900 break-all">{episodeId ?? 'Episode Detail'}</h1>
          {episode && (
            <div className="text-sm text-gray-600 space-x-2">
              <span title={formatDate(episode.created_at)}>Created: {formatRelativeTime(episode.created_at)}</span>
              {episode.eval_task_id && <span>Eval Task: {episode.eval_task_id}</span>}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          {episode?.primary_pv_id ? (
            <Link
              to={`/leaderboard/policy/${episode.primary_pv_id}`}
              className="inline-flex items-center px-3 py-2 rounded border border-blue-500 text-blue-600 no-underline hover:bg-blue-50 text-sm"
            >
              View Policy
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
            <h2 className="text-lg font-semibold text-gray-900">Summary</h2>
          </div>
        </div>
        <div className="p-5 space-y-4">
          {state.loading ? (
            <div className="text-gray-500 text-sm">Loading episode...</div>
          ) : state.error ? (
            <div className="text-red-600 text-sm">{state.error}</div>
          ) : episode ? (
            <>
              <div className="flex flex-wrap gap-4 text-sm text-gray-800">
                <div>
                  <span className="font-semibold">Episode ID:</span>{' '}
                  <span className="font-mono break-all">{episode.id}</span>
                </div>
                <div>
                  <span className="font-semibold">Primary Policy:</span>{' '}
                  {episode.primary_pv_id ? (
                    <Link
                      to={`/leaderboard/policy/${episode.primary_pv_id}`}
                      className="text-blue-600 no-underline hover:underline font-mono"
                    >
                      {episode.primary_pv_id}
                    </Link>
                  ) : (
                    '—'
                  )}
                </div>
                <div>
                  <span className="font-semibold">Avg Reward:</span> {formatScore(episode.avg_reward)}
                </div>
                <div title={formatDate(episode.created_at)}>
                  <span className="font-semibold">Created:</span> {formatRelativeTime(episode.created_at)}
                </div>
                <div>
                  <span className="font-semibold">Replay:</span>{' '}
                  {episode.replay_url ? (
                    <a
                      href={episode.replay_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 no-underline hover:underline"
                    >
                      Open
                    </a>
                  ) : (
                    '—'
                  )}
                </div>
              </div>

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
