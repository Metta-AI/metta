import { FC, Fragment, useContext, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { AppContext } from './AppContext'
import { ReplayViewer, normalizeReplayUrl } from './components/ReplayViewer'
import {
  LEADERBOARD_ATTEMPTS_TAG,
  LEADERBOARD_DONE_TAG,
  LEADERBOARD_EVAL_CANCELED_VALUE,
  LEADERBOARD_EVAL_DONE_VALUE,
  LEADERBOARD_SIM_NAME_EPISODE_KEY,
} from './constants'
import type { EpisodeReplay, LeaderboardPolicyEntry } from './repo'
import { formatPolicyVersion } from './utils/format'

type SectionState = {
  entries: LeaderboardPolicyEntry[]
  loading: boolean
  error: string | null
}

type ViewKey = 'public'

type ViewConfig = {
  sectionKey: ViewKey
  label: string
  subtitle: string
  emptyMessage: string
}

type EvalStatusInfo = {
  attempts: number | null
  status: 'pending' | 'complete' | 'canceled'
  label: string
}

type ReplayFetchState = {
  loading: boolean
  error: string | null
  episodesBySimulation: Record<string, EpisodeReplay[]>
}

const REFRESH_INTERVAL_MS = 10_000

const STYLES = `
.leaderboard-page {
  padding: 24px;
  min-height: calc(100vh - 60px);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #fff;
  color: #0f172a;
}

.leaderboard-card {
  max-width: 1000px;
  margin: 0 auto;
  background: #fff;
  border: 1px solid #e0e7ff;
  border-radius: 10px;
  padding: 20px 24px 28px;
  box-shadow: 0 4px 10px rgba(15, 23, 42, 0.05);
}

.leaderboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 16px;
}

.leaderboard-title {
  margin: 0;
  font-size: 24px;
}

.leaderboard-toggle {
  display: inline-flex;
  gap: 8px;
}

.toggle-button {
  border: 1px solid #cbd5f5;
  border-radius: 4px;
  background: #fff;
  padding: 6px 12px;
  font-size: 14px;
  cursor: pointer;
}

.toggle-button.active {
  background: #1d4ed8;
  color: #fff;
  border-color: #1d4ed8;
}

.section-state {
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  padding: 16px;
  font-size: 14px;
  color: #475569;
  text-align: center;
}

.section-state.error {
  border-color: #fecaca;
  color: #b91c1c;
}

.leaderboard-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.leaderboard-table th {
  text-align: left;
  padding: 8px 4px;
  border-bottom: 1px solid #e2e8f0;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #475569;
  background: #f9fafc;
}

.leaderboard-table td {
  border-bottom: 1px solid #f1f5f9;
  padding: 10px 4px;
  vertical-align: top;
}

.policy-row {
  cursor: pointer;
  transition: background 0.15s ease;
}

.policy-row:hover {
  background: #f8fafc;
}

.policy-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.policy-link {
  color: #1d4ed8;
  text-decoration: none;
}

.policy-link:hover {
  text-decoration: underline;
}

.policy-title-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.policy-title-row .policy-title {
  margin-bottom: 0;
  flex: 1;
}

.policy-link {
  color: inherit;
  text-decoration: none;
}

.policy-link:hover {
  text-decoration: underline;
}

.policy-link-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  border: 1px solid #cbd5f5;
  border-radius: 4px;
  background: #fff;
  color: #1d4ed8;
  padding: 4px 8px;
  font-size: 12px;
  text-decoration: none;
}

.policy-link-button:hover {
  background: #eff6ff;
}

.policy-status-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
  margin-left: auto;
}

.policy-status-badge.pending {
  background: #f1f5f9;
  color: #475569;
  border-color: #cbd5f5;
}

.policy-status-badge.complete {
  background: #dcfce7;
  color: #15803d;
  border-color: #86efac;
}

.policy-status-badge.canceled {
  background: #fee2e2;
  color: #b91c1c;
  border-color: #fecaca;
}

.policy-meta {
  font-size: 12px;
  color: #64748b;
}

.policy-details-row td {
  background: #f8fafc;
  border-bottom: none;
}

.policy-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 12px 0;
}

.detail-block {
  font-size: 14px;
}

.detail-heading {
  margin: 0 0 4px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #475569;
}

.score-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.score-table th,
.score-table td {
  border: 1px solid #e2e8f0;
  padding: 6px;
}

.score-table th {
  background: #f3f4f6;
  color: #475569;
  font-weight: 600;
}

.replay-links {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.replay-link {
  color: #1d4ed8;
  text-decoration: none;
  font-weight: 600;
  font-size: 12px;
}

.replay-link:hover {
  text-decoration: underline;
}

.replay-button {
  border: 1px solid #cbd5f5;
  border-radius: 4px;
  background: #fff;
  padding: 4px 8px;
  font-size: 12px;
  color: #1d4ed8;
  cursor: pointer;
}

.replay-button:hover {
  background: #eff6ff;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}

.tag {
  border: 1px solid #cbd5f5;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 12px;
  background: #eef2ff;
  color: #1d4ed8;
}

.metadata-inline {
  font-size: 13px;
  color: #0f172a;
  word-break: break-word;
}

.command-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.command-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.command-item-label {
  font-size: 12px;
  font-weight: 600;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.copy-command {
  border: 1px solid #cbd5f5;
  border-radius: 6px;
  padding: 8px 12px;
  background: #f8fafc;
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  text-align: left;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.copy-command:hover {
  border-color: #94a3b8;
}

.copy-command.copied {
  border-color: #1d4ed8;
  background: #eef2ff;
}

.copy-command code {
  flex: 1;
  font-size: 12px;
  color: #0f172a;
  word-break: break-word;
}

.copy-command-status {
  font-size: 12px;
  font-weight: 600;
  color: #1d4ed8;
  white-space: nowrap;
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(15, 23, 42, 0.2);
  border-top-color: #1d4ed8;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 8px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
`

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

const formatScore = (value: number | null): string => {
  if (typeof value !== 'number') {
    return '—'
  }
  return value.toFixed(2)
}

const formatSimulationLabel = (identifier: string): string => {
  const parts = identifier.split(':')
  return parts[parts.length - 1] || identifier
}

const parseSimulationTag = (
  identifier: string
): {
  tagKey: string | null
  tagValue: string | null
} => {
  const separatorIndex = identifier.indexOf(':')
  if (separatorIndex === -1) {
    return { tagKey: null, tagValue: null }
  }
  return {
    tagKey: identifier.slice(0, separatorIndex),
    tagValue: identifier.slice(separatorIndex + 1),
  }
}

const getEvalStatus = (tags: Record<string, string>): EvalStatusInfo => {
  const attemptValue = tags[LEADERBOARD_ATTEMPTS_TAG]
  const parsedAttempts = attemptValue !== undefined ? Number(attemptValue) : null
  const attempts = typeof parsedAttempts === 'number' && Number.isFinite(parsedAttempts) ? parsedAttempts : null
  const doneValue = tags[LEADERBOARD_DONE_TAG]
  if (doneValue === LEADERBOARD_EVAL_CANCELED_VALUE) {
    return { attempts, status: 'canceled', label: 'Canceled' }
  }
  if (doneValue === LEADERBOARD_EVAL_DONE_VALUE) {
    return { attempts, status: 'complete', label: 'Complete' }
  }
  return { attempts, status: 'pending', label: 'Pending' }
}

const buildReplayUrl = (replayUrl: string | null | undefined): string | null =>
  normalizeReplayUrl(replayUrl) ?? replayUrl ?? null

const createInitialSectionState = (): SectionState => ({
  entries: [],
  loading: true,
  error: null,
})

export const Leaderboard: FC = () => {
  const { repo } = useContext(AppContext)
  const [publicLeaderboard, setPublicLeaderboard] = useState<SectionState>(() => createInitialSectionState())
  const [expandedRows, setExpandedRows] = useState<Set<string>>(() => new Set())
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null)
  const [replayState, setReplayState] = useState<Record<string, ReplayFetchState>>({})
  const [replayPreviews, setReplayPreviews] = useState<Record<string, { url: string; label: string }>>({})

  const viewConfig: ViewConfig = {
    sectionKey: 'public',
    label: 'Public',
    subtitle: 'Published policies submitted to the cogames leaderboard.',
    emptyMessage: 'No public leaderboard entries yet.',
  }

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setPublicLeaderboard((prev) => ({ ...prev, loading: prev.entries.length === 0, error: null }))
      try {
        // Use the new endpoint that returns entries with VOR already computed
        const response = await repo.getPublicLeaderboard()
        if (!ignore) {
          setPublicLeaderboard({ entries: response.entries, loading: false, error: null })
        }
      } catch (error: any) {
        if (!ignore) {
          setPublicLeaderboard({ entries: [], loading: false, error: error.message ?? 'Failed to load leaderboard' })
        }
      }
    }
    load()
    const intervalId =
      typeof window !== 'undefined' ? window.setInterval(() => void load(), REFRESH_INTERVAL_MS) : undefined
    return () => {
      ignore = true
      if (typeof window !== 'undefined' && intervalId !== undefined) {
        window.clearInterval(intervalId)
      }
    }
  }, [repo])

  const toggleRow = (rowKey: string, entry: LeaderboardPolicyEntry) => {
    setExpandedRows((previous) => {
      const next = new Set(previous)
      if (next.has(rowKey)) {
        next.delete(rowKey)
      } else {
        next.add(rowKey)
        void fetchReplaysForPolicy(entry)
      }
      return next
    })
  }

  const copyCommandToClipboard = async (command: string, label: string): Promise<void> => {
    if (typeof navigator === 'undefined' || typeof navigator.clipboard === 'undefined') {
      return
    }
    try {
      await navigator.clipboard.writeText(command)
      setCopiedCommand(label)
      if (typeof window !== 'undefined') {
        window.setTimeout(() => {
          setCopiedCommand((previous) => (previous === label ? null : previous))
        }, 2000)
      }
    } catch (error) {
      console.error('Failed to copy command to clipboard:', error)
    }
  }

  const renderCopyableCommand = (command: string, commandKey: string, label: string) => (
    <button
      type="button"
      className={`copy-command ${copiedCommand === commandKey ? 'copied' : ''}`}
      onClick={() => void copyCommandToClipboard(command, commandKey)}
    >
      <code>{command}</code>
      <span className="copy-command-status">{copiedCommand === commandKey ? 'Copied!' : label}</span>
    </button>
  )

  const fetchReplaysForPolicy = async (entry: LeaderboardPolicyEntry) => {
    const policyId = entry.policy_version.id
    const existing = replayState[policyId]
    if (existing && (existing.loading || Object.keys(existing.episodesBySimulation).length > 0 || existing.error)) {
      return
    }

    const simTagValues = Array.from(
      new Set(
        Object.keys(entry.scores)
          .map((simKey) => parseSimulationTag(simKey))
          .filter((parsed) => parsed.tagKey === LEADERBOARD_SIM_NAME_EPISODE_KEY && parsed.tagValue)
          .map((parsed) => parsed.tagValue as string)
      )
    )

    if (simTagValues.length === 0) {
      setReplayState((prev) => ({
        ...prev,
        [policyId]: { loading: false, error: null, episodesBySimulation: {} },
      }))
      return
    }

    setReplayState((prev) => ({
      ...prev,
      [policyId]: { loading: true, error: null, episodesBySimulation: {} },
    }))

    try {
      const response = await repo.queryEpisodes({
        primary_policy_version_ids: [policyId],
        tag_filters: { [LEADERBOARD_SIM_NAME_EPISODE_KEY]: simTagValues },
        limit: null,
      })

      const episodesBySimulation: Record<string, EpisodeReplay[]> = {}
      response.episodes.forEach((episode) => {
        const tagValue = episode.tags?.[LEADERBOARD_SIM_NAME_EPISODE_KEY]
        if (!tagValue || !episode.replay_url) {
          return
        }
        const simKey = `${LEADERBOARD_SIM_NAME_EPISODE_KEY}:${tagValue}`
        const expectedEpisodeId = entry.score_episode_ids?.[simKey]
        if (expectedEpisodeId && expectedEpisodeId !== episode.id) {
          return
        }
        episodesBySimulation[simKey] = [
          ...(episodesBySimulation[simKey] ?? []),
          { replay_url: episode.replay_url, episode_id: episode.id },
        ]
      })

      setReplayState((prev) => ({
        ...prev,
        [policyId]: { loading: false, error: null, episodesBySimulation },
      }))
    } catch (error: any) {
      setReplayState((prev) => ({
        ...prev,
        [policyId]: { loading: false, error: error.message ?? 'Failed to load replays', episodesBySimulation: {} },
      }))
    }
  }

  const toggleReplayPreview = (policyId: string, label: string, replayUrl: string | null | undefined) => {
    const normalized = buildReplayUrl(replayUrl)
    if (!normalized) {
      return
    }
    setReplayPreviews((prev) => {
      const existing = prev[policyId]
      if (existing?.url === normalized) {
        const next = { ...prev }
        delete next[policyId]
        return next
      }
      return { ...prev, [policyId]: { url: normalized, label } }
    })
  }

  const renderContent = (state: SectionState, config: ViewConfig) => {
    if (state.loading) {
      return (
        <div className="section-state">
          <div className="loading-spinner" />
          Loading policies...
        </div>
      )
    }
    if (state.error) {
      return <div className="section-state error">{state.error}</div>
    }
    if (state.entries.length === 0) {
      return <div className="section-state">{config.emptyMessage}</div>
    }

    return (
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th>Policy</th>
            <th>Policy Created</th>
            <th>Avg score</th>
          </tr>
        </thead>
        <tbody>
          {state.entries.map((entry) => {
            const { policy_version: policyVersion } = entry
            const policyId = policyVersion.id
            const policyDisplay = formatPolicyVersion(policyVersion)
            const createdAt = policyVersion.policy_created_at || policyVersion.created_at
            const rowKey = `${config.sectionKey}-${policyId}`
            const isExpanded = expandedRows.has(rowKey)
            const scoreEntries = Object.entries(entry.scores).sort(([a], [b]) => a.localeCompare(b))
            const tagEntries = Object.entries(policyVersion.tags).sort(([a], [b]) => a.localeCompare(b))
            const evalStatus = getEvalStatus(policyVersion.tags)
            const policyUri = `metta://policy/${policyVersion.name}:v${policyVersion.version}`
            const evaluateCommand = `./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=${policyId}`
            const playCommand = `./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=${policyId}`
            const replayPreview = replayPreviews[policyId]
            return (
              <Fragment key={`${config.sectionKey}-${policyId}`}>
                <tr className="policy-row" onClick={() => toggleRow(rowKey, entry)}>
                  <td>
                    <div className="policy-title-row">
                      <div className="policy-title">{policyDisplay}</div>
                      <Link
                        to={`/policies/versions/${policyId}`}
                        className="policy-link-button"
                        onClick={(event) => event.stopPropagation()}
                      >
                        {'View Details'}
                      </Link>
                      <span className={`policy-status-badge ${evalStatus.status}`}>{evalStatus.label}</span>
                    </div>
                    <div className="policy-meta">{policyVersion.user_id}</div>
                  </td>
                  <td>
                    <div className="policy-meta">{formatDate(createdAt)}</div>
                  </td>
                  <td>
                    <div className="policy-title">{formatScore(entry.avg_score ?? null)}</div>
                  </td>
                </tr>
                {isExpanded && (
                  <tr className="policy-details-row">
                    <td colSpan={3}>
                      <div className="policy-details">
                        <div className="detail-block">
                          {scoreEntries.length === 0 ? (
                            <div className="policy-meta">No simulation scores available.</div>
                          ) : (
                            <table className="score-table">
                              <thead>
                                <tr>
                                  <th>Simulation</th>
                                  <th>Score</th>
                                  <th>Replays</th>
                                </tr>
                              </thead>
                              <tbody>
                                {scoreEntries.map(([simName, scoreValue]) => {
                                  const replaysForPolicy = replayState[policyId]
                                  const simReplays = replaysForPolicy?.episodesBySimulation[simName] ?? []
                                  const isLoadingReplays = replaysForPolicy?.loading
                                  const replayError = replaysForPolicy?.error
                                  return (
                                    <tr key={`${policyId}-${simName}`}>
                                      <td>{formatSimulationLabel(simName)}</td>
                                      <td>{scoreValue.toFixed(2)}</td>
                                      <td>
                                        {replayError ? (
                                          <span className="policy-meta">Failed to load replays</span>
                                        ) : isLoadingReplays ? (
                                          <span className="policy-meta">Loading replays...</span>
                                        ) : simReplays.length === 0 ? (
                                          '—'
                                        ) : (
                                          <div className="replay-links">
                                            {simReplays.map((replay, replayIndex) => {
                                              const label =
                                                replay.episode_id && replay.episode_id.length > 0
                                                  ? `Episode ${replay.episode_id.slice(0, 8)}`
                                                  : `Replay ${replayIndex + 1}`
                                              const replayUrl = buildReplayUrl(replay.replay_url)
                                              if (!replayUrl) {
                                                return null
                                              }
                                              return (
                                                <div
                                                  key={`${simName}-${replay.episode_id}-${replayIndex}`}
                                                  className="flex items-center gap-2"
                                                >
                                                  <a
                                                    href={replayUrl}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="replay-link"
                                                    onClick={(event) => event.stopPropagation()}
                                                  >
                                                    {label}
                                                  </a>
                                                  <button
                                                    type="button"
                                                    className="replay-button"
                                                    onClick={(event) => {
                                                      event.stopPropagation()
                                                      toggleReplayPreview(
                                                        policyId,
                                                        `${formatSimulationLabel(simName)} • ${label}`,
                                                        replay.replay_url
                                                      )
                                                    }}
                                                  >
                                                    Show below
                                                  </button>
                                                </div>
                                              )
                                            })}
                                          </div>
                                        )}
                                      </td>
                                    </tr>
                                  )
                                })}
                              </tbody>
                            </table>
                          )}
                        </div>
                        {replayPreview ? (
                          <div className="detail-block">
                            <div className="detail-heading">Replay Preview</div>
                            <ReplayViewer replayUrl={replayPreview.url} label={replayPreview.label} />
                          </div>
                        ) : null}
                        {evalStatus.status === 'canceled' && (
                          <div className="detail-block">
                            <div className="detail-heading">Leaderboard Eval Status</div>
                            <div className="policy-meta">
                              Evaluations were automatically canceled after repeated failures.
                            </div>
                          </div>
                        )}
                        <div className="detail-block">
                          <div className="command-list">
                            <div className="command-item">
                              <div className="command-item-label">Policy URI</div>
                              {renderCopyableCommand(policyUri, `${policyId}-uri`, 'Copy')}
                            </div>
                            <div className="command-item">
                              <div className="command-item-label">Evaluate</div>
                              {renderCopyableCommand(evaluateCommand, `${policyId}-evaluate`, 'Copy')}
                            </div>
                            <div className="command-item">
                              <div className="command-item-label">Play</div>
                              {renderCopyableCommand(playCommand, `${policyId}-play`, 'Copy')}
                            </div>
                          </div>
                        </div>
                        <div className="detail-block">
                          {tagEntries.length === 0 ? null : (
                            <div className="tag-list">
                              {tagEntries.map(([key, value]) => (
                                <span key={`${policyId}-${key}-${value}`} className="tag">
                                  {key}: {value}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    )
  }

  return (
    <div className="leaderboard-page">
      <style>{STYLES}</style>
      <div className="leaderboard-card">
        <div className="leaderboard-header">
          <div>
            <h1 className="leaderboard-title">Leaderboard</h1>
          </div>
        </div>
        {renderContent(publicLeaderboard, viewConfig)}
      </div>
    </div>
  )
}
