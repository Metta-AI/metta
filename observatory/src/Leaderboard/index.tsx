import clsx from 'clsx'
import { FC, PropsWithChildren, useContext, useEffect, useState } from 'react'

import { AppContext } from '../AppContext'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { Table, TH, TR } from '../components/Table'
import type { LeaderboardPolicyEntry } from '../repo'
import { LeaderboardEntry } from './LeaderboardEntry'

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

const REFRESH_INTERVAL_MS = 10_000

const LoadingBox: FC<PropsWithChildren<{ error?: boolean }>> = ({ children, error }) => {
  return (
    <div className={clsx('border border-gray-300 rounded-md p-4 text-center text-sm', error && 'border-red-300')}>
      {children}
    </div>
  )
}

const createInitialSectionState = (): SectionState => ({
  entries: [],
  loading: true,
  error: null,
})

export const Leaderboard: FC = () => {
  const { repo } = useContext(AppContext)
  const [publicLeaderboard, setPublicLeaderboard] = useState<SectionState>(() => createInitialSectionState())

  const viewConfig: ViewConfig = {
    sectionKey: 'public',
    label: 'Public',
    subtitle: 'Published policies submitted to the cogames leaderboard.',
    emptyMessage: 'No public leaderboard entries yet.',
  }

  useEffect(() => {
    const load = async () => {
      setPublicLeaderboard((prev) => ({ ...prev, loading: prev.entries.length === 0, error: null }))
      try {
        // Use the new endpoint that returns entries with VOR already computed
        const response = await repo.getPublicLeaderboard()
        setPublicLeaderboard({ entries: response.entries, loading: false, error: null })
      } catch (error: any) {
        setPublicLeaderboard({ entries: [], loading: false, error: error.message ?? 'Failed to load leaderboard' })
      }
    }
    load()
    const intervalId = window.setInterval(() => load(), REFRESH_INTERVAL_MS)
    return () => {
      window.clearInterval(intervalId)
    }
  }, [repo])

  const renderContent = (state: SectionState, config: ViewConfig) => {
    if (state.loading) {
      return (
        <LoadingBox>
          <div className="flex flex-col gap-1 items-center">
            <Spinner size="lg" />
            Loading policies...
          </div>
        </LoadingBox>
      )
    }
    if (state.error) {
      return <LoadingBox error>{state.error}</LoadingBox>
    }
    if (state.entries.length === 0) {
      return <LoadingBox>{config.emptyMessage}</LoadingBox>
    }

    return (
      <Table>
        <Table.Header>
          <TR>
            <TH>Policy</TH>
            <TH>Policy Created</TH>
            <TH>Avg score</TH>
          </TR>
        </Table.Header>
        <Table.Body>
          {state.entries.map((entry) => (
            <LeaderboardEntry key={entry.policy_version.id} entry={entry} />
          ))}
        </Table.Body>
      </Table>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <Card title="Leaderboard">{renderContent(publicLeaderboard, viewConfig)}</Card>
    </div>
  )
}
