import { FC, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { AppContext } from '../AppContext'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { StyledLink } from '../components/StyledLink'
import { Table, TH, TR, TD } from '../components/Table'
import { PlayerDetail } from '../repo'
import { formatRelativeTime } from '../utils/datetime'

const ActionBadge: FC<{ action: string }> = ({ action }) => {
  const colors: Record<string, string> = {
    add: 'bg-green-100 text-green-800',
    remove: 'bg-gray-100 text-gray-600',
  }
  return <span className={`px-2 py-1 rounded text-xs font-medium ${colors[action] || 'bg-gray-100'}`}>{action}</span>
}

export const PlayerPage: FC = () => {
  const { seasonName, policyVersionId } = useParams<{ seasonName: string; policyVersionId: string }>()
  const { repo } = useContext(AppContext)

  const [player, setPlayer] = useState<PlayerDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let ignore = false
    const load = async () => {
      if (!seasonName || !policyVersionId) return
      try {
        const data = await repo.getSeasonPlayer(seasonName, policyVersionId)
        if (!ignore) {
          setPlayer(data)
          setError(null)
        }
      } catch (err: any) {
        if (!ignore) {
          setError(err.message)
        }
      } finally {
        if (!ignore) {
          setLoading(false)
        }
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [repo, seasonName, policyVersionId])

  if (loading) {
    return (
      <div className="p-6 max-w-5xl mx-auto flex justify-center py-16">
        <Spinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <Card>
          <div className="text-red-600">{error}</div>
        </Card>
      </div>
    )
  }

  if (!player) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <Card>
          <div className="text-gray-500">Player not found</div>
        </Card>
      </div>
    )
  }

  const playerName =
    player.policy.name && player.policy.version !== null
      ? `${player.policy.name}:v${player.policy.version}`
      : player.policy.id.slice(0, 8)

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/tournament" className="text-blue-500 hover:text-blue-700">
          Tournament
        </Link>
        <span className="text-gray-400">/</span>
        <Link to={`/tournament/${seasonName}`} className="text-blue-500 hover:text-blue-700">
          {seasonName}
        </Link>
        <span className="text-gray-400">/</span>
        <h1 className="text-2xl font-bold text-gray-900 font-mono">{playerName}</h1>
      </div>

      <Card title="Policy">
        <div className="space-y-2">
          <div>
            <span className="text-gray-500 text-sm">Policy version: </span>
            <StyledLink to={`/policies/versions/${player.policy.id}`} className="font-mono text-sm">
              {playerName}
            </StyledLink>
          </div>
        </div>
      </Card>

      <Card title="Membership History">
        {player.membership_history.length === 0 ? (
          <div className="text-gray-500 py-4">No membership changes recorded</div>
        ) : (
          <Table>
            <Table.Header>
              <TH>Time</TH>
              <TH>Pool</TH>
              <TH>Action</TH>
              <TH>Notes</TH>
            </Table.Header>
            <Table.Body>
              {player.membership_history.map((entry, i) => (
                <TR key={i}>
                  <TD className="text-gray-500 text-sm">{formatRelativeTime(entry.created_at)}</TD>
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
    </div>
  )
}
