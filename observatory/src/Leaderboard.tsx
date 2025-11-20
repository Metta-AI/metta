import { Fragment, useEffect, useState } from 'react'

import type { LeaderboardPolicyEntry, Repo } from './repo'

type LeaderboardProps = {
  repo: Repo
  currentUser: string
}

type SectionState = {
  entries: LeaderboardPolicyEntry[]
  loading: boolean
  error: string | null
}

type ViewKey = 'public' | 'mine'

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
const LEADERBOARD_SIM_VERSION = 'v0.1'
const LEADERBOARD_ATTEMPTS_TAG = `leaderboard-attempts-${LEADERBOARD_SIM_VERSION}`
const LEADERBOARD_DONE_TAG = `leaderboard-evals-done-${LEADERBOARD_SIM_VERSION}`
const LEADERBOARD_DONE_VALUE = 'true'
const LEADERBOARD_CANCELED_VALUE = 'canceled'

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

const getEvalStatus = (tags: Record<string, string>): EvalStatusInfo => {
  const attemptValue = tags[LEADERBOARD_ATTEMPTS_TAG]
  const parsedAttempts = attemptValue !== undefined ? Number(attemptValue) : null
  const attempts = typeof parsedAttempts === 'number' && Number.isFinite(parsedAttempts) ? parsedAttempts : null
  const doneValue = tags[LEADERBOARD_DONE_TAG]
  if (doneValue === LEADERBOARD_CANCELED_VALUE) {
    return { attempts, status: 'canceled', label: 'Canceled' }
  }
  if (doneValue === LEADERBOARD_DONE_VALUE) {
    return { attempts, status: 'complete', label: 'Complete' }
  }
  return { attempts, status: 'pending', label: 'Pending' }
}

const createInitialSectionState = (): SectionState => ({
  entries: [],
  loading: true,
  error: null,
})

export function Leaderboard({ repo, currentUser }: LeaderboardProps) {
  const [publicLeaderboard, setPublicLeaderboard] = useState<SectionState>(() => createInitialSectionState())
  const [personalLeaderboard, setPersonalLeaderboard] = useState<SectionState>(() => createInitialSectionState())
  const [view, setView] = useState<ViewKey>('public')
  const [expandedRows, setExpandedRows] = useState<Set<string>>(() => new Set())
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null)

  const viewConfigs: Record<ViewKey, ViewConfig> = {
    public: {
      sectionKey: 'public',
      label: 'Public',
      subtitle: 'Published policies submitted to the cogames leaderboard.',
      emptyMessage: 'No public leaderboard entries yet.',
    },
    mine: {
      sectionKey: 'mine',
      label: 'Mine',
      subtitle: `Entries uploaded by ${currentUser}.`,
      emptyMessage: 'You have not submitted any leaderboard policies yet.',
    },
  }
  const toggleOptions: ViewKey[] = ['public', 'mine']
  const activeState = view === 'public' ? publicLeaderboard : personalLeaderboard
  const activeConfig = viewConfigs[view]

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setPublicLeaderboard((prev) => ({ ...prev, loading: prev.entries.length === 0, error: null }))
      try {
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

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setPersonalLeaderboard((prev) => ({ ...prev, loading: prev.entries.length === 0, error: null }))
      try {
        const response = await repo.getPersonalLeaderboard()
        if (!ignore) {
          setPersonalLeaderboard({ entries: response.entries, loading: false, error: null })
        }
      } catch (error: any) {
        if (!ignore) {
          setPersonalLeaderboard({ entries: [], loading: false, error: error.message ?? 'Failed to load leaderboard' })
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

  const toggleRow = (rowKey: string) => {
    setExpandedRows((previous) => {
      const next = new Set(previous)
      if (next.has(rowKey)) {
        next.delete(rowKey)
      } else {
        next.add(rowKey)
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
            <th>Avg Score</th>
          </tr>
        </thead>
        <tbody>
          {state.entries.map((entry) => {
            const { policy_version: policyVersion } = entry
            const policyId = policyVersion.id
            const policyDisplay = `${policyVersion.name}.${policyVersion.version}`
            const createdAt = policyVersion.policy_created_at || policyVersion.created_at
            const rowKey = `${config.sectionKey}-${policyId}`
            const isExpanded = expandedRows.has(rowKey)
            const scoreEntries = Object.entries(entry.scores).sort(([a], [b]) => a.localeCompare(b))
            const tagEntries = Object.entries(policyVersion.tags).sort(([a], [b]) => a.localeCompare(b))
            const evalStatus = getEvalStatus(policyVersion.tags)
            const evaluateCommand = `./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=${policyId}`
            const playCommand = `./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=${policyId}`
            return (
              <Fragment key={`${config.sectionKey}-${policyId}`}>
                <tr className="policy-row" onClick={() => toggleRow(rowKey)}>
                  <td>
                    <div className="policy-title-row">
                      <div className="policy-title">{policyDisplay}</div>
                      <span className={`policy-status-badge ${evalStatus.status}`}>{evalStatus.label}</span>
                    </div>
                    <div className="policy-meta">{policyVersion.user_id}</div>
                  </td>
                  <td>
                    <div className="policy-meta">{formatDate(createdAt)}</div>
                  </td>
                  <td>
                    <div className="policy-title">{formatScore(entry.avg_score)}</div>
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
                                </tr>
                              </thead>
                              <tbody>
                                {scoreEntries.map(([simName, scoreValue]) => (
                                  <tr key={`${policyId}-${simName}`}>
                                    <td>{formatSimulationLabel(simName)}</td>
                                    <td>{scoreValue.toFixed(2)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          )}
                        </div>
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
                              {renderCopyableCommand(evaluateCommand, `${policyId}-evaluate`, 'Evaluate')}
                            </div>
                            <div className="command-item">
                              {renderCopyableCommand(playCommand, `${policyId}-play`, 'Play')}
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
          <div className="leaderboard-toggle">
            {toggleOptions.map((option) => {
              const config = viewConfigs[option]
              return (
                <button
                  key={config.sectionKey}
                  className={`toggle-button ${view === option ? 'active' : ''}`}
                  onClick={() => setView(option)}
                >
                  {config.label}
                </button>
              )
            })}
          </div>
        </div>
        {renderContent(activeState, activeConfig)}
      </div>
    </div>
  )
}
