import { FC, useContext, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { Card } from './components/Card'
import { ReplayViewer } from './components/ReplayViewer'
import { SmallHeader } from './components/SmallHeader'
import { StyledLink } from './components/StyledLink'
import { Table, TD, TH, TR } from './components/Table'
import { TagList } from './components/TagList'
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
      </div>

      <Card title="Policies & Scores">
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
            <Table>
              <Table.Header>
                <TH>Policy</TH>
                <TH>Policy ID</TH>
                <TH>Avg Reward</TH>
              </Table.Header>
              <Table.Body>
                {Object.entries(episode.avg_rewards || {})
                  .sort(([, a], [, b]) => (typeof b === 'number' && typeof a === 'number' ? b - a : 0))
                  .map(([policyId, reward]) => {
                    const info = policyInfoState.data[policyId]
                    const policyLabel = formatPolicyVersion(info)
                    return (
                      <TR key={policyId}>
                        <TD>
                          <div className="flex items-center gap-2">
                            <StyledLink to={`/policies/versions/${policyId}`}>{policyLabel}</StyledLink>
                            {policyInfoState.loading && !info ? (
                              <span className="text-xs text-gray-400">(loading details)</span>
                            ) : null}
                          </div>
                        </TD>
                        <TD>
                          <span className="font-mono text-xs text-gray-700">{policyId}</span>
                        </TD>
                        <TD>
                          <span className="font-mono">{formatScore(reward)}</span>
                        </TD>
                      </TR>
                    )
                  })}
              </Table.Body>
            </Table>
            {policyInfoState.error ? (
              <div className="text-red-600 text-xs mt-2">Failed to load policy details: {policyInfoState.error}</div>
            ) : null}
          </div>
        )}
      </Card>

      <Card title="Replay">
        {state.loading ? (
          <div className="text-gray-500 text-sm">Loading replay...</div>
        ) : state.error ? (
          <div className="text-red-600 text-sm">{state.error}</div>
        ) : !episode || !episode.replay_url ? (
          <div className="text-gray-500 text-sm">No replay available for this episode.</div>
        ) : (
          <ReplayViewer replayUrl={episode.replay_url} label="Episode replay" />
        )}
      </Card>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="p-5 space-y-6">
          {state.loading ? (
            <div className="text-gray-500 text-sm">Loading episode...</div>
          ) : state.error ? (
            <div className="text-red-600 text-sm">{state.error}</div>
          ) : episode ? (
            <>
              <div>
                <SmallHeader>Tags</SmallHeader>
                {Object.keys(episode.tags).length === 0 ? (
                  <div className="text-gray-500 text-sm">No tags found.</div>
                ) : (
                  <TagList tags={episode.tags} />
                )}
              </div>

              <div>
                <SmallHeader>Attributes</SmallHeader>
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
