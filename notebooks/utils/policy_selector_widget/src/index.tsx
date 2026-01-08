import React, { useEffect } from 'react'
import { useModelState, createRender } from '@anywidget/react'
import PolicySelector from './PolicySelector'
import { PolicyInfo, FilterState, UIConfig } from './types'
import './styles.css'

function PolicySelectorWidget() {
  // Use anywidget React bridge hooks for state management
  const [policyData] = useModelState<PolicyInfo[]>('policy_data')
  const [selectedPolicies, setSelectedPolicies] = useModelState<string[]>('selected_policies')
  const [searchTerm, setSearchTerm] = useModelState<string>('search_term')
  const [policyTypeFilter, setPolicyTypeFilter] = useModelState<string[]>('policy_type_filter')
  const [tagFilter, setTagFilter] = useModelState<string[]>('tag_filter')
  const [useApiSearch] = useModelState<boolean>('use_api_search')
  const [searchDebounceMs] = useModelState<number>('search_debounce_ms')
  const [searchCompleted] = useModelState<any>('api_search_completed')

  // Event traits for communication with Python
  const [, setSelectionChanged] = useModelState<any>('selection_changed')
  const [, setFilterChanged] = useModelState<any>('filter_changed')
  const [, setSearchTrigger] = useModelState<number>('search_trigger')
  const [, setCurrentSearchParams] = useModelState<any>('current_search_params')
  const [, setLoadAllPoliciesRequested] = useModelState<boolean>('load_all_policies_requested')

  // UI configuration
  const uiConfig: UIConfig = {
    showTags: true,
    showType: true,
    showCreatedAt: true,
    maxDisplayedPolicies: 100,
  }

  const handleSelectionChange = (selectedIds: string[]) => {
    setSelectedPolicies(selectedIds)
    setSelectionChanged({
      selected_policies: selectedIds,
      action: 'selection_updated',
      timestamp: Date.now(),
    })
  }

  const handleFilterChange = (filter: FilterState) => {
    // Update the actual filter state
    setSearchTerm(filter.searchTerm || '')
    setPolicyTypeFilter(filter.policyTypeFilter || [])
    setTagFilter(filter.tagFilter || [])

    // Also send the filter change event to Python
    setFilterChanged({
      ...filter,
      timestamp: Date.now(),
    })
  }

  const handleApiSearch = (filter: FilterState) => {
    console.log(`🚀 React sending API search request to Python:`, filter)

    const searchRequest = {
      ...filter,
      timestamp: Date.now(),
    }

    // Update local state so dropdowns show selected values
    handleFilterChange(filter)

    // Use the counter-based approach that we know works
    setCurrentSearchParams(searchRequest)
    setSearchTrigger((prev) => (prev || 0) + 1)

    console.log(`🔢 Set search_trigger and current_search_params`)
  }

  useEffect(() => {
    // Load all policies when component mounts
    console.log('🚀 Loading all policies from client on mount')
    setLoadAllPoliciesRequested(true)
  }, [setLoadAllPoliciesRequested])

  return (
    <PolicySelector
      policies={policyData || []}
      selectedPolicies={selectedPolicies || []}
      searchTerm={searchTerm || ''}
      policyTypeFilter={policyTypeFilter || []}
      tagFilter={tagFilter || []}
      uiConfig={uiConfig}
      useApiSearch={useApiSearch || false}
      searchDebounceMs={searchDebounceMs || 300}
      searchCompleted={searchCompleted}
      onSelectionChange={handleSelectionChange}
      onFilterChange={handleFilterChange}
      onApiSearch={handleApiSearch}
    />
  )
}

export default {
  render: createRender(PolicySelectorWidget),
}
