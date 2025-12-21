import { FC, Fragment, useContext, useState } from 'react'

import { AppContext } from '../AppContext'
import { Button } from '../components/Button'
import { CommandItem } from '../components/CommandItem'
import { LinkButton } from '../components/LinkButton'
import { normalizeReplayUrl, ReplayViewer } from '../components/ReplayViewer'
import { SmallHeader } from '../components/SmallHeader'
import { Spinner } from '../components/Spinner'
import { Table, TD, TH, TR } from '../components/Table'
import { TagList } from '../components/TagList'
import {
  LEADERBOARD_ATTEMPTS_TAG,
  LEADERBOARD_DONE_TAG,
  LEADERBOARD_EVAL_CANCELED_VALUE,
  LEADERBOARD_EVAL_DONE_VALUE,
  LEADERBOARD_SIM_NAME_EPISODE_KEY,
} from '../constants'
import { EpisodeReplay, LeaderboardPolicyEntry } from '../repo'
import { formatPolicyVersion } from '../utils/format'
import { PolicyStatusBadge } from './PolicyStatusBadge'

type ReplayFetchState = {
  loading: boolean
  error: string | null
  episodesBySimulation: Record<string, EpisodeReplay[]>
}

function formatSimulationLabel(identifier: string): string {
  const parts = identifier.split(':')
  return parts[parts.length - 1] || identifier
}

function buildReplayUrl(replayUrl: string | null | undefined): string | null {
  return normalizeReplayUrl(replayUrl) ?? replayUrl ?? null
}

function parseSimulationTag(identifier: string): {
  tagKey: string | null
  tagValue: string | null
} {
  const separatorIndex = identifier.indexOf(':')
  if (separatorIndex === -1) {
    return { tagKey: null, tagValue: null }
  }
  return {
    tagKey: identifier.slice(0, separatorIndex),
    tagValue: identifier.slice(separatorIndex + 1),
  }
}

export type EvalStatusInfo = {
  attempts: number | null
  status: 'pending' | 'complete' | 'canceled'
}

function formatDate(value: string | null): string {
  if (!value) {
    return '—'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString()
}

function formatScore(value: number | null): string {
  if (typeof value !== 'number') {
    return '—'
  }
  return value.toFixed(2)
}

function getEvalStatus(tags: Record<string, string>): EvalStatusInfo {
  const attemptValue = tags[LEADERBOARD_ATTEMPTS_TAG]
  const parsedAttempts = attemptValue !== undefined ? Number(attemptValue) : null
  const attempts = typeof parsedAttempts === 'number' && Number.isFinite(parsedAttempts) ? parsedAttempts : null
  const doneValue = tags[LEADERBOARD_DONE_TAG]
  if (doneValue === LEADERBOARD_EVAL_CANCELED_VALUE) {
    return { attempts, status: 'canceled' }
  }
  if (doneValue === LEADERBOARD_EVAL_DONE_VALUE) {
    return { attempts, status: 'complete' }
  }
  return { attempts, status: 'pending' }
}

export const LeaderboardEntry: FC<{ entry: LeaderboardPolicyEntry }> = ({ entry }) => {
  const { repo } = useContext(AppContext)

  const [isExpanded, setIsExpanded] = useState(false)

  const { policy_version: policyVersion } = entry
  const policyId = policyVersion.id
  const policyDisplay = formatPolicyVersion(policyVersion)
  const createdAt = policyVersion.policy_created_at || policyVersion.created_at
  const evalStatus = getEvalStatus(policyVersion.tags)
  const policyUri = `metta://policy/${policyVersion.name}:v${policyVersion.version}`
  const evaluateCommand = `./tools/run.py recipes.experiment.v0_leaderboard.evaluate policy_version_id=${policyId}`
  const playCommand = `./tools/run.py recipes.experiment.v0_leaderboard.play policy_version_id=${policyId}`

  const [replayPreview, setReplayPreview] = useState<{ url: string; label: string } | null>(null)
  const [replayState, setReplayState] = useState<ReplayFetchState | null>(null)

  const scoreEntries = Object.entries(entry.scores).sort(([a], [b]) => a.localeCompare(b))

  const toggleReplayPreview = (label: string, replayUrl: string | null | undefined) => {
    const normalized = buildReplayUrl(replayUrl)
    if (!normalized) {
      return
    }
    if (replayPreview?.url === normalized) {
      setReplayPreview(null)
    } else {
      setReplayPreview({ url: normalized, label })
    }
  }

  const fetchReplays = async () => {
    const policyId = entry.policy_version.id
    if (
      replayState &&
      (replayState.loading || Object.keys(replayState.episodesBySimulation).length > 0 || replayState.error)
    ) {
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
      setReplayState({ loading: false, error: null, episodesBySimulation: {} })
      return
    }

    setReplayState({ loading: true, error: null, episodesBySimulation: {} })

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

      setReplayState({ loading: false, error: null, episodesBySimulation })
    } catch (error: any) {
      setReplayState({ loading: false, error: error.message ?? 'Failed to load replays', episodesBySimulation: {} })
    }
  }

  const toggleExpanded = () => {
    // fetch replays when the table is expanded for the first time
    if (!isExpanded) {
      fetchReplays()
    }
    setIsExpanded(!isExpanded)
  }

  return (
    <Fragment>
      <TR className="cursor-pointer hover:bg-gray-50 transition-colors duration-150" onClick={toggleExpanded}>
        <TD>
          <div className="flex items-center gap-2 justify-between">
            <div className="font-semibold">{policyDisplay}</div>
            <div className="flex items-center gap-2">
              <LinkButton to={`/policies/versions/${policyId}`} size="sm">
                View Details
              </LinkButton>
              <PolicyStatusBadge status={evalStatus} />
            </div>
          </div>
          <div className="text-xs text-gray-500">{policyVersion.user_id}</div>
        </TD>
        <TD>
          <div className="text-xs text-gray-500">{formatDate(createdAt)}</div>
        </TD>
        <TD>
          <div className="font-semibold">{formatScore(entry.avg_score ?? null)}</div>
        </TD>
      </TR>
      {isExpanded && (
        <TR className="bg-gray-50">
          <TD colSpan={3}>
            <div className="py-3 flex flex-col gap-3">
              <div>
                {scoreEntries.length === 0 ? (
                  <div className="text-xs text-gray-600 my-2">No simulation scores available.</div>
                ) : (
                  <Table theme="inner">
                    <Table.Header>
                      <TR>
                        <TH>Simulation</TH>
                        <TH>Score</TH>
                        <TH>Replays</TH>
                      </TR>
                    </Table.Header>
                    <Table.Body>
                      {scoreEntries.map(([simName, scoreValue]) => {
                        const simReplays = replayState?.episodesBySimulation[simName] ?? []
                        const isLoadingReplays = replayState?.loading
                        const replayError = replayState?.error
                        return (
                          <TR key={simName}>
                            <TD>{formatSimulationLabel(simName)}</TD>
                            <TD>{scoreValue.toFixed(2)}</TD>
                            <TD>
                              {replayError ? (
                                <span className="text-red-700">Failed to load replays</span>
                              ) : isLoadingReplays ? (
                                <Spinner />
                              ) : simReplays.length === 0 ? (
                                '—'
                              ) : (
                                <div className="flex flex-wrap gap-">
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
                                          className="no-underline font-bold text-xs"
                                          onClick={(event) => event.stopPropagation()}
                                        >
                                          {label}
                                        </a>
                                        <Button
                                          size="sm"
                                          onClick={() => {
                                            toggleReplayPreview(
                                              `${formatSimulationLabel(simName)} • ${label}`,
                                              replay.replay_url
                                            )
                                          }}
                                        >
                                          Show below
                                        </Button>
                                      </div>
                                    )
                                  })}
                                </div>
                              )}
                            </TD>
                          </TR>
                        )
                      })}
                    </Table.Body>
                  </Table>
                )}
                {replayPreview ? (
                  <div>
                    <SmallHeader>Replay Preview</SmallHeader>
                    <ReplayViewer replayUrl={replayPreview.url} label={replayPreview.label} />
                  </div>
                ) : null}
              </div>
              {evalStatus.status === 'canceled' && (
                <div>
                  <SmallHeader>Leaderboard Eval Status</SmallHeader>
                  <div className="text-xs text-gray-500">
                    Evaluations were automatically canceled after repeated failures.
                  </div>
                </div>
              )}
              <div>
                <div className="flex flex-col gap-2">
                  <CommandItem label="Policy URI" command={policyUri} />
                  <CommandItem label="Evaluate" command={evaluateCommand} />
                  <CommandItem label="Play" command={playCommand} />
                </div>
              </div>
              <TagList tags={policyVersion.tags} />
            </div>
          </TD>
        </TR>
      )}
    </Fragment>
  )
}
