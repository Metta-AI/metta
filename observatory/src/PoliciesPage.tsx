import { FC, useContext, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import { AppContext } from './AppContext'
import type { PolicyRow } from './repo'
import { formatDate, formatRelativeTime } from './utils/datetime'

const DEBOUNCE_MS = 300

type LoadState<T> = {
  data: T
  loading: boolean
  error: string | null
}

export const PoliciesPage: FC = () => {
  const { repo } = useContext(AppContext)
  const [policiesState, setPoliciesState] = useState<LoadState<PolicyRow[]>>({
    data: [],
    loading: true,
    error: null,
  })
  const [nameFilter, setNameFilter] = useState('')
  const [debouncedFilter, setDebouncedFilter] = useState('')
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }
    debounceRef.current = setTimeout(() => {
      setDebouncedFilter(nameFilter)
    }, DEBOUNCE_MS)

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [nameFilter])

  useEffect(() => {
    let isMounted = true

    const loadPolicies = async () => {
      setPoliciesState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const response = await repo.getPolicies({
          limit: 500,
          name_fuzzy: debouncedFilter || undefined,
        })
        if (isMounted) {
          setPoliciesState({ data: response.entries, loading: false, error: null })
        }
      } catch (error: any) {
        if (isMounted) {
          setPoliciesState({
            data: [],
            loading: false,
            error: error.message ?? 'Failed to load policies',
          })
        }
      }
    }

    void loadPolicies()

    return () => {
      isMounted = false
    }
  }, [repo, debouncedFilter])

  const filteredPolicies = policiesState.data.filter((policy) =>
    policy.name.toLowerCase().includes(nameFilter.toLowerCase())
  )

  const isInitialLoad = policiesState.loading && policiesState.data.length === 0
  const isSearching = policiesState.loading && policiesState.data.length > 0

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-gray-900">Policies</h1>
          <p className="text-sm text-gray-500">All policies ordered by creation date</p>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center gap-3">
          <input
            type="text"
            placeholder="Search by name..."
            value={nameFilter}
            onChange={(e) => setNameFilter(e.target.value)}
            className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          {isSearching && (
            <svg
              className="animate-spin h-4 w-4 text-gray-400"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          )}
        </div>
        <div className="p-5">
          {isInitialLoad ? (
            <div className="text-gray-500 text-sm">Loading policies...</div>
          ) : policiesState.error ? (
            <div className="text-red-600 text-sm">{policiesState.error}</div>
          ) : filteredPolicies.length === 0 ? (
            <div className="text-gray-500 text-sm">No policies found.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-50 text-left text-xs font-semibold uppercase text-gray-600">
                    <th className="px-3 py-2 border-b border-gray-200">Name</th>
                    <th className="px-3 py-2 border-b border-gray-200">Versions</th>
                    <th className="px-3 py-2 border-b border-gray-200">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPolicies.map((policy) => (
                    <tr key={policy.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-3 py-2">
                        <Link
                          to={`/policies/${policy.id}`}
                          className="text-blue-600 no-underline hover:underline font-medium"
                        >
                          {policy.name}
                        </Link>
                      </td>
                      <td className="px-3 py-2">
                        <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-gray-100 border border-gray-200">
                          {policy.version_count} version{policy.version_count !== 1 ? 's' : ''}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-gray-600" title={formatDate(policy.created_at)}>
                        {formatRelativeTime(policy.created_at)}
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
