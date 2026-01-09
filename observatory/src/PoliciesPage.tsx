import { FC, useContext, useEffect, useState } from 'react'

import { AppContext } from './AppContext'
import { Input } from './components/Input'
import { Spinner } from './components/Spinner'
import { StyledLink } from './components/StyledLink'
import { Table, TD, TH, TR } from './components/Table'
import { useDebouncedValue } from './hooks/useDebouncedValue'
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
  const debouncedFilter = useDebouncedValue(nameFilter, DEBOUNCE_MS)

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

    loadPolicies()

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
          <div className="w-full max-w-xs">
            <Input placeholder="Search by name..." value={nameFilter} onChange={(value) => setNameFilter(value)} />
          </div>
          {isSearching && <Spinner />}
        </div>
        <div className="p-5">
          {isInitialLoad ? (
            <Spinner size="lg" />
          ) : policiesState.error ? (
            <div className="text-red-600 text-sm">{policiesState.error}</div>
          ) : filteredPolicies.length === 0 ? (
            <div className="text-gray-500 text-sm">No policies found.</div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <Table.Header>
                  <TH>Name</TH>
                  <TH>Versions</TH>
                  <TH>Created</TH>
                </Table.Header>
                <Table.Body>
                  {filteredPolicies.map((policy) => (
                    <TR key={policy.id}>
                      <TD>
                        <StyledLink to={`/policies/${policy.id}`}>{policy.name}</StyledLink>
                      </TD>
                      <TD>
                        <span className="inline-flex items-center px-2 py-1 text-xs rounded bg-gray-100 border border-gray-200 text-nowrap">
                          {policy.version_count} version{policy.version_count !== 1 ? 's' : ''}
                        </span>
                      </TD>
                      <TD title={formatDate(policy.created_at)}>{formatRelativeTime(policy.created_at)}</TD>
                    </TR>
                  ))}
                </Table.Body>
              </Table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
