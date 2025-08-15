import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { PolicyInfo, FilterState, UIConfig } from './types'

interface PolicySelectorProps {
  policies: PolicyInfo[]
  selectedPolicies: string[]
  searchTerm: string
  policyTypeFilter: string[]
  tagFilter: string[]
  uiConfig: UIConfig
  useApiSearch: boolean
  searchDebounceMs: number
  searchCompleted?: any
  onSelectionChange: (selectedIds: string[]) => void
  onFilterChange: (filter: FilterState) => void
  onApiSearch?: (filters: FilterState) => void
}

const PolicySelector: React.FC<PolicySelectorProps> = ({
  policies,
  selectedPolicies,
  searchTerm,
  policyTypeFilter,
  tagFilter,
  uiConfig,
  useApiSearch,
  searchDebounceMs,
  searchCompleted,
  onSelectionChange,
  onFilterChange,
  onApiSearch,
}) => {
  const [localSearchTerm, setLocalSearchTerm] = useState<string>(searchTerm)
  const [isSearching, setIsSearching] = useState<boolean>(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debug component mount and key props (remove in production)
  // useEffect(() => {
  //   console.log(`🎯 PolicySelector mounted/updated:`, {
  //     useApiSearch,
  //     hasOnApiSearch: !!onApiSearch,
  //     policiesCount: policies.length,
  //     searchTerm,
  //     localSearchTerm
  //   });
  // }, [useApiSearch, onApiSearch, policies.length, searchTerm, localSearchTerm]);

  // Filter and sort policies
  const filteredPolicies = useMemo(() => {
    // If using API search, trust the policies prop as-is (it contains API results)
    if (useApiSearch) {
      return policies.slice(0, uiConfig.maxDisplayedPolicies)
    }

    // Only do client-side filtering when NOT using API search
    let filtered = policies

    // Filter by search term
    if (localSearchTerm) {
      const search = localSearchTerm.toLowerCase()
      filtered = filtered.filter((policy) => policy.name.toLowerCase().includes(search))
    }

    // Filter by policy type
    if (policyTypeFilter.length > 0) {
      filtered = filtered.filter((policy) => policyTypeFilter.includes(policy.type))
    }

    // Filter by tags
    if (tagFilter.length > 0) {
      filtered = filtered.filter((policy) => tagFilter.some((tag) => policy.tags.includes(tag)))
    }

    // Limit displayed policies
    return filtered.slice(0, uiConfig.maxDisplayedPolicies)
  }, [policies, localSearchTerm, policyTypeFilter, tagFilter, uiConfig.maxDisplayedPolicies, useApiSearch])

  // Get unique policy types and tags for filter dropdowns
  const availableTypes = useMemo(() => {
    const types = new Set(policies.map((p) => p.type))
    return Array.from(types).sort()
  }, [policies])

  const availableTags = useMemo(() => {
    const tags = new Set(policies.flatMap((p) => p.tags))
    return Array.from(tags).sort()
  }, [policies])

  // Debounced search function
  const debouncedSearch = useCallback(
    (searchValue: string) => {
      const filters: FilterState = {
        searchTerm: searchValue,
        policyTypeFilter,
        tagFilter,
      }

      if (useApiSearch && onApiSearch) {
        setIsSearching(true)
        onApiSearch(filters)
      } else {
        // For local filtering, never show spinner
        setIsSearching(false)
        onFilterChange(filters)
      }
    },
    [policyTypeFilter, tagFilter, useApiSearch, onApiSearch, onFilterChange]
  )

  const handleSearchChange = (value: string) => {
    setLocalSearchTerm(value)

    // Clear existing debounce timer
    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }

    if (useApiSearch && onApiSearch) {
      // Always use debounced API search when client is configured
      debounceRef.current = setTimeout(() => {
        debouncedSearch(value)
      }, searchDebounceMs)
    } else {
      // Immediate local filtering when no client
      onFilterChange({ searchTerm: value, policyTypeFilter, tagFilter })
    }
  }

  // Clean up debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [])

  // Update isSearching state when policies change (search completed)
  useEffect(() => {
    if (isSearching) {
      setIsSearching(false)
    }
  }, [policies])

  // Listen for explicit search completion signal
  useEffect(() => {
    if (searchCompleted && isSearching) {
      console.log(`✅ Search completed signal received, stopping spinner`)
      setIsSearching(false)
    }
  }, [searchCompleted, isSearching])

  // Add a timeout to prevent spinner from spinning indefinitely
  useEffect(() => {
    if (isSearching) {
      const timeout = setTimeout(() => {
        console.warn('Search timeout - resetting spinner (no response from API)')
        setIsSearching(false)
      }, 5000) // 5 second timeout

      return () => clearTimeout(timeout)
    }
    return undefined
  }, [isSearching])

  // Also reset search state when useApiSearch changes to false
  useEffect(() => {
    if (!useApiSearch && isSearching) {
      setIsSearching(false)
    }
  }, [useApiSearch, isSearching])

  const handleTypeFilterChange = (types: string[]) => {
    onFilterChange({ searchTerm: localSearchTerm, policyTypeFilter: types, tagFilter })
  }

  const handleTagFilterChange = (tags: string[]) => {
    onFilterChange({ searchTerm: localSearchTerm, policyTypeFilter, tagFilter: tags })
  }

  const handlePolicyToggle = (policyId: string) => {
    const newSelection = selectedPolicies.includes(policyId)
      ? selectedPolicies.filter((id) => id !== policyId)
      : [...selectedPolicies, policyId]
    onSelectionChange(newSelection)
  }

  const handleSelectAll = () => {
    const allIds = filteredPolicies.map((p) => p.id)
    onSelectionChange([...new Set([...selectedPolicies, ...allIds])])
  }

  const handleClearSelection = () => {
    onSelectionChange([])
  }

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString()
    } catch {
      return dateString
    }
  }

  return (
    <div className="policy-selector-widget">
      <div className="policy-selector-header">
        <h3 className="policy-selector-title">Policy Selector</h3>
      </div>

      <div className="policy-selector-search">
        <div className="search-input-container">
          <input
            type="text"
            className={`search-input ${isSearching ? 'searching' : ''}`}
            placeholder={useApiSearch ? 'Search policies (API)...' : 'Search policies by name...'}
            value={localSearchTerm}
            onChange={(e) => handleSearchChange(e.target.value)}
          />
          {isSearching && <div className="search-loading-spinner" />}
        </div>

        {availableTypes.length > 1 && (
          <select
            className="filter-dropdown"
            value={policyTypeFilter.length > 0 ? policyTypeFilter[0] : ''}
            onChange={(e) => {
              const value = e.target.value
              handleTypeFilterChange(value ? [value] : [])

              // Trigger immediate filter application
              if (useApiSearch && onApiSearch) {
                const filters: FilterState = {
                  searchTerm: localSearchTerm,
                  policyTypeFilter: value ? [value] : [],
                  tagFilter,
                }
                onApiSearch(filters)
              }
            }}
          >
            <option value="">All Types</option>
            {availableTypes.map((type) => (
              <option key={type} value={type}>
                {type === 'training_run' ? 'Training Run' : 'Policy'}
              </option>
            ))}
          </select>
        )}

        {availableTags.length > 0 && (
          <select
            className="filter-dropdown"
            value={tagFilter.length > 0 ? tagFilter[0] : ''}
            onChange={(e) => {
              const value = e.target.value
              handleTagFilterChange(value ? [value] : [])

              // Trigger immediate filter application
              if (useApiSearch && onApiSearch) {
                const filters: FilterState = {
                  searchTerm: localSearchTerm,
                  policyTypeFilter,
                  tagFilter: value ? [value] : [],
                }
                onApiSearch(filters)
              }
            }}
          >
            <option value="">All Tags</option>
            {availableTags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="selection-controls">
        <button className="selection-button" onClick={handleSelectAll}>
          Select All Filtered
        </button>
        <button className="selection-button" onClick={handleClearSelection}>
          Clear Selection
        </button>
        <div className="selection-count">
          {selectedPolicies.length} selected, {filteredPolicies.length} shown
        </div>
      </div>

      <div className="policy-list">
        {filteredPolicies.length === 0 ? (
          <div className="policy-list-empty">
            <div className="policy-list-empty-icon">🔍</div>
            <div>{policies.length === 0 ? 'No policies available' : 'No policies match your filters'}</div>
          </div>
        ) : (
          filteredPolicies.map((policy) => (
            <div
              key={policy.id}
              className={`policy-item ${selectedPolicies.includes(policy.id) ? 'selected' : ''}`}
              onClick={() => handlePolicyToggle(policy.id)}
            >
              <input
                type="checkbox"
                className="policy-checkbox"
                checked={selectedPolicies.includes(policy.id)}
                onChange={() => handlePolicyToggle(policy.id)}
                onClick={(e) => e.stopPropagation()}
              />

              <div className="policy-info">
                <div className="policy-name">{policy.name}</div>

                <div className="policy-meta">
                  {uiConfig.showType && (
                    <span className={`policy-type ${policy.type}`}>
                      {policy.type === 'training_run' ? 'Training Run' : 'Policy'}
                    </span>
                  )}

                  {uiConfig.showTags && policy.tags.length > 0 && (
                    <div className="policy-tags">
                      {policy.tags.map((tag) => (
                        <span key={tag} className="policy-tag">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {uiConfig.showCreatedAt && <span className="policy-created-at">{formatDate(policy.created_at)}</span>}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default PolicySelector
