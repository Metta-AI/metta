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

type SectionConfig = {
  title: string
  subtitle: string
  state: SectionState
  emptyMessage: string
  sectionKey: string
}

const STYLES = `
.leaderboard-page {
  min-height: calc(100vh - 60px);
  padding: 32px 24px 48px;
  background: #f6f7fb;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.leaderboard-header {
  max-width: 1100px;
  margin: 0 auto 24px;
}

.leaderboard-header h1 {
  font-size: 28px;
  color: #0f172a;
  margin: 0 0 6px;
}

.leaderboard-header p {
  color: #475569;
  margin: 0;
  font-size: 15px;
}

.leaderboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 24px;
  max-width: 1100px;
  margin: 0 auto;
}

.leaderboard-section {
  background: #fff;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  padding: 20px;
  box-shadow: 0 10px 15px rgba(15, 23, 42, 0.03);
}

.leaderboard-section h2 {
  font-size: 20px;
  color: #0f172a;
  margin: 0;
}

.leaderboard-section p {
  color: #475569;
  margin: 4px 0 16px;
  font-size: 14px;
}

.section-state {
  border: 1px dashed #cbd5f5;
  border-radius: 12px;
  padding: 24px;
  text-align: center;
  color: #475569;
  font-size: 14px;
}

.section-state.error {
  border-color: #fecdd3;
  color: #be123c;
  background: #fef2f2;
}

.leaderboard-table {
  width: 100%;
  border-collapse: collapse;
}

.leaderboard-table thead th {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
  border-bottom: 1px solid #e2e8f0;
  padding: 10px 6px;
  text-align: left;
}

.policy-row {
  cursor: pointer;
  transition: background 0.15s ease;
}

.policy-row:hover {
  background: #f8fafc;
}

.policy-row td {
  padding: 12px 6px;
  border-bottom: 1px solid #f1f5f9;
}

.policy-title {
  font-size: 16px;
  font-weight: 600;
  color: #0f172a;
}

.policy-meta {
  font-size: 12px;
  color: #64748b;
}

.policy-details-row td {
  padding: 16px;
  background: #f8fafc;
}

.policy-details {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.detail-section h4 {
  margin: 0 0 8px;
  font-size: 13px;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.score-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
}

.score-pill {
  border-radius: 10px;
  background: #fff;
  border: 1px solid #cbd5f5;
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.score-pill span:first-of-type {
  font-size: 12px;
  color: #475569;
}

.score-pill span:last-of-type {
  font-size: 15px;
  color: #0f172a;
  font-weight: 600;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.tag-pill {
  background: #eef2ff;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  color: #3730a3;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(29, 78, 216, 0.2);
  border-top-color: #1d4ed8;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 8px auto;
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

const createInitialSectionState = (): SectionState => ({
  entries: [],
  loading: true,
  error: null,
})

export function Leaderboard({ repo, currentUser }: LeaderboardProps) {
  const [publicLeaderboard, setPublicLeaderboard] = useState<SectionState>(() => createInitialSectionState())
  const [personalLeaderboard, setPersonalLeaderboard] = useState<SectionState>(() => createInitialSectionState())
  const [expandedRows, setExpandedRows] = useState<Set<string>>(() => new Set())

  useEffect(() => {
    let ignore = false
    const load = async () => {
      setPublicLeaderboard((prev) => ({ ...prev, loading: true, error: null }))
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
    return () => {
      ignore = true
    }
  }, [repo])

  useEffect(() => {
    let ignore = false
    const load = async () => {
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
    return () => {
      ignore = true
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

  const renderSection = ({ title, subtitle, state, emptyMessage, sectionKey }: SectionConfig) => (
    <section key={sectionKey} className="leaderboard-section">
      <div className="section-header">
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {state.loading ? (
        <div className="section-state">
          <div className="loading-spinner" />
          Loading policies...
        </div>
      ) : state.error ? (
        <div className="section-state error">{state.error}</div>
      ) : state.entries.length === 0 ? (
        <div className="section-state">{emptyMessage}</div>
      ) : (
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
              const rowKey = `${sectionKey}-${policyId}`
              const isExpanded = expandedRows.has(rowKey)
              const scoreEntries = Object.entries(entry.scores).sort(([a], [b]) => a.localeCompare(b))
              const tagEntries = Object.entries(policyVersion.tags).sort(([a], [b]) => a.localeCompare(b))
              return (
                <Fragment key={`${sectionKey}-${policyId}`}>
                  <tr className="policy-row" onClick={() => toggleRow(rowKey)}>
                    <td>
                      <div className="policy-title">{policyDisplay}</div>
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
                          <div className="detail-section">
                            <h4>Scores by Simulation</h4>
                            {scoreEntries.length === 0 ? (
                              <div className="policy-meta">No simulation scores available.</div>
                            ) : (
                              <div className="score-grid">
                                {scoreEntries.map(([simName, scoreValue]) => (
                                  <div className="score-pill" key={`${policyId}-${simName}`}>
                                    <span>{simName}</span>
                                    <span>{scoreValue.toFixed(2)}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                          <div className="detail-section">
                            <h4>Tags</h4>
                            {tagEntries.length === 0 ? (
                              <div className="policy-meta">No tags for this policy version.</div>
                            ) : (
                              <div className="tag-list">
                                {tagEntries.map(([key, value]) => (
                                  <span key={`${policyId}-${key}-${value}`} className="tag-pill">
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
      )}
    </section>
  )

  const sections: SectionConfig[] = [
    {
      title: 'Public leaderboard',
      subtitle: 'Published policies submitted to the cogames leaderboard.',
      state: publicLeaderboard,
      emptyMessage: 'No public leaderboard entries yet.',
      sectionKey: 'public',
    },
    {
      title: 'My submissions',
      subtitle: `Entries uploaded by ${currentUser}.`,
      state: personalLeaderboard,
      emptyMessage: 'You have not submitted any leaderboard policies yet.',
      sectionKey: 'mine',
    },
  ]

  return (
    <div className="leaderboard-page">
      <style>{STYLES}</style>
      <div className="leaderboard-header">
        <h1>Leaderboard</h1>
      </div>
      <div className="leaderboard-grid">{sections.map((section) => renderSection(section))}</div>
    </div>
  )
}
