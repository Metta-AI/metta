import { FC, useContext, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { Card } from './components/Card'
import { CopyableUri } from './components/CopyableUri'
import { LinkButton } from './components/LinkButton'
import { Spinner } from './components/Spinner'
import { StyledLink } from './components/StyledLink'
import { Table, TD, TH, TR } from './components/Table'
import type { PublicPolicyVersionRow } from './repo'
import { formatDate, formatRelativeTime } from './utils/datetime'

type LoadState<T> = {
  data: T
  loading: boolean
  error: string | null
}

export const PolicyPage: FC = () => {
  const { policyId } = useParams<{ policyId: string }>()
  const { repo } = useContext(AppContext)

  const [versionsState, setVersionsState] = useState<LoadState<PublicPolicyVersionRow[]>>({
    data: [],
    loading: true,
    error: null,
  })

  useEffect(() => {
    if (!policyId) {
      setVersionsState({ data: [], loading: false, error: 'Missing policy id' })
      return
    }

    let isMounted = true

    const loadVersions = async () => {
      setVersionsState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const response = await repo.getVersionsForPolicy(policyId, { limit: 500 })
        if (isMounted) {
          setVersionsState({ data: response.entries, loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setVersionsState({
            data: [],
            loading: false,
            error: error.message ?? 'Failed to load policy versions',
          })
        }
      }
    }

    loadVersions()

    return () => {
      isMounted = false
    }
  }, [policyId, repo])

  const policyName = versionsState.loading ? 'Loading...' : (versionsState.data[0]?.name ?? 'Unknown Policy')
  const policyCreatedAt = versionsState.data[0]?.policy_created_at ?? null
  const userId = versionsState.data[0]?.user_id

  useEffect(() => {
    const title = versionsState.loading ? 'Loading...' : (versionsState.data[0]?.name ?? 'Policy')
    document.title = `${title} | Observatory`
  }, [versionsState.loading, versionsState.data])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Policy</p>
          <h1 className="text-2xl font-semibold text-gray-900">{policyName}</h1>
          {!versionsState.loading && (
            <div className="flex flex-wrap gap-3 text-sm text-gray-600">
              {userId && <span className="text-gray-500">User: {userId}</span>}
              {policyCreatedAt && (
                <span className="text-gray-500" title={formatDate(policyCreatedAt)}>
                  Created: {formatRelativeTime(policyCreatedAt)}
                </span>
              )}
              <span className="flex items-center gap-1 text-gray-500">
                Policy ID:
                <span className="font-mono text-xs text-gray-700">{policyId}</span>
              </span>
            </div>
          )}
        </div>
        <LinkButton to="/" theme="tertiary">
          ← Back to policies
        </LinkButton>
      </div>

      {!versionsState.loading && versionsState.data[0]?.name && (
        <CopyableUri uri={`metta://policy/${versionsState.data[0].name}`} />
      )}

      <Card title="Versions">
        {versionsState.loading ? (
          <Spinner />
        ) : versionsState.error ? (
          <div className="text-red-600 text-sm">{versionsState.error}</div>
        ) : versionsState.data.length === 0 ? (
          <div className="text-gray-500 text-sm">No versions found for this policy.</div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <Table.Header>
                <TR>
                  <TH>Version</TH>
                  <TH>Version ID</TH>
                  <TH>Created</TH>
                </TR>
              </Table.Header>
              <Table.Body>
                {versionsState.data.map((pv) => (
                  <TR key={pv.id}>
                    <TD>
                      <StyledLink to={`/policies/versions/${pv.id}`}>v{pv.version}</StyledLink>
                    </TD>
                    <TD>
                      <span className="font-mono text-xs text-gray-600">{pv.id}</span>
                    </TD>
                    <TD title={formatDate(pv.created_at)}>{formatRelativeTime(pv.created_at)}</TD>
                  </TR>
                ))}
              </Table.Body>
            </Table>
          </div>
        )}
      </Card>
    </div>
  )
}
