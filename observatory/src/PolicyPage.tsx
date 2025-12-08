import { FC, useContext, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { AppContext } from './AppContext'
import { CopyableUri } from './components/CopyableUri'
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

    void loadVersions()

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
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="inline-flex items-center px-3 py-2 rounded border border-gray-300 text-gray-700 no-underline hover:bg-gray-50 text-sm"
          >
            ← Back to policies
          </Link>
        </div>
      </div>

      {!versionsState.loading && versionsState.data[0]?.name && (
        <CopyableUri uri={`metta://policy/${versionsState.data[0].name}`} />
      )}

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="px-5 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Versions</h2>
        </div>
        <div className="p-5">
          {versionsState.loading ? (
            <div className="text-gray-500 text-sm">Loading versions...</div>
          ) : versionsState.error ? (
            <div className="text-red-600 text-sm">{versionsState.error}</div>
          ) : versionsState.data.length === 0 ? (
            <div className="text-gray-500 text-sm">No versions found for this policy.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-50 text-left text-xs font-semibold uppercase text-gray-600">
                    <th className="px-3 py-2 border-b border-gray-200">Version</th>
                    <th className="px-3 py-2 border-b border-gray-200">Version ID</th>
                    <th className="px-3 py-2 border-b border-gray-200">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {versionsState.data.map((pv) => (
                    <tr key={pv.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-3 py-2">
                        <Link
                          to={`/policies/versions/${pv.id}`}
                          className="text-blue-600 no-underline hover:underline font-medium"
                        >
                          v{pv.version}
                        </Link>
                      </td>
                      <td className="px-3 py-2">
                        <span className="font-mono text-xs text-gray-600">{pv.id}</span>
                      </td>
                      <td className="px-3 py-2 text-gray-600" title={formatDate(pv.created_at)}>
                        {formatRelativeTime(pv.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
