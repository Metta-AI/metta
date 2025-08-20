import React, { useState } from 'react'
import { UnifiedPolicyInfo } from '../repo'
import { SearchInput } from './SearchInput'
import styles from './PolicySelector.module.css'

interface PolicySelectorProps {
  policies: UnifiedPolicyInfo[]
  selectedTrainingRunIds: string[]
  selectedRunFreePolicyIds: string[]
  onTrainingRunSelectionChange: (selectedIds: string[]) => void
  onRunFreePolicySelectionChange: (selectedIds: string[]) => void
  pageSize?: number
}

export const PolicySelector: React.FC<PolicySelectorProps> = React.memo(
  ({
    policies,
    selectedTrainingRunIds,
    selectedRunFreePolicyIds,
    onTrainingRunSelectionChange,
    onRunFreePolicySelectionChange,
    pageSize = 25,
  }) => {
    const [currentPage, setCurrentPage] = useState(1)
    const [searchText, setSearchText] = useState('')

    // Helper functions
    const formatDate = (dateStr: string) => {
      try {
        return new Date(dateStr).toLocaleDateString()
      } catch {
        return dateStr
      }
    }

    // Client-side filtering and pagination
    const filtered = policies.filter((policy) =>
      policy.name.toLowerCase().includes(searchText.toLowerCase()) ||
      (policy.user_id && policy.user_id.toLowerCase().includes(searchText.toLowerCase())) ||
      policy.tags.some(tag => tag.toLowerCase().includes(searchText.toLowerCase()))
    )

    const startIndex = (currentPage - 1) * pageSize
    const endIndex = startIndex + pageSize
    const filteredAndPaginatedPolicies = filtered.slice(startIndex, endIndex)

    // Calculate total pages for client-side pagination
    const totalPages = Math.ceil(filtered.length / pageSize)

    // Check if all visible policies are selected
    const allPoliciesSelected = filtered.length > 0 &&
      filtered.every((policy) => {
        if (policy.type === 'training_run') {
          return selectedTrainingRunIds.includes(policy.id)
        } else {
          return selectedRunFreePolicyIds.includes(policy.id)
        }
      })

    const handleSearchChange = (newSearchText: string) => {
      setSearchText(newSearchText)
      // Reset to first page when search text changes
      if (currentPage !== 1) {
        setCurrentPage(1)
      }
    }

    const handlePolicyToggle = (policyId: string, policyType: string) => {
      if (policyType === 'training_run') {
        const newSelection = selectedTrainingRunIds.includes(policyId)
          ? selectedTrainingRunIds.filter((id) => id !== policyId)
          : [...selectedTrainingRunIds, policyId]
        onTrainingRunSelectionChange(newSelection)
      } else {
        const newSelection = selectedRunFreePolicyIds.includes(policyId)
          ? selectedRunFreePolicyIds.filter((id) => id !== policyId)
          : [...selectedRunFreePolicyIds, policyId]
        onRunFreePolicySelectionChange(newSelection)
      }
    }

    const handleSelectAllPolicies = () => {
      if (allPoliciesSelected) {
        // Deselect all filtered policies
        const visibleTrainingRunIds = filtered
          .filter(p => p.type === 'training_run')
          .map(p => p.id)
        const visiblePolicyIds = filtered
          .filter(p => p.type === 'policy')
          .map(p => p.id)

        const newTrainingRunSelection = selectedTrainingRunIds.filter(id => !visibleTrainingRunIds.includes(id))
        const newPolicySelection = selectedRunFreePolicyIds.filter(id => !visiblePolicyIds.includes(id))

        onTrainingRunSelectionChange(newTrainingRunSelection)
        onRunFreePolicySelectionChange(newPolicySelection)
      } else {
        // Select all visible policies
        const visibleTrainingRunIds = filtered
          .filter(p => p.type === 'training_run')
          .map(p => p.id)
        const visiblePolicyIds = filtered
          .filter(p => p.type === 'policy')
          .map(p => p.id)

        const newTrainingRunSelection = [...new Set([...selectedTrainingRunIds, ...visibleTrainingRunIds])]
        const newPolicySelection = [...new Set([...selectedRunFreePolicyIds, ...visiblePolicyIds])]

        onTrainingRunSelectionChange(newTrainingRunSelection)
        onRunFreePolicySelectionChange(newPolicySelection)
      }
    }

    const handleClearAll = () => {
      onTrainingRunSelectionChange([])
      onRunFreePolicySelectionChange([])
    }

    const isPolicySelected = (policyId: string, policyType: string) => {
      if (policyType === 'training_run') {
        return selectedTrainingRunIds.includes(policyId)
      } else {
        return selectedRunFreePolicyIds.includes(policyId)
      }
    }

    return (
      <div className={styles.container}>
        <SearchInput searchText={searchText} onSearchChange={handleSearchChange} disabled={false} />

        <div className={styles.policiesContainer}>
          {filteredAndPaginatedPolicies.length === 0 && (
            <div className={styles.emptyMessage}>
              {searchText
                ? 'No policies match your search.'
                : 'No policies available.'}
            </div>
          )}

          {filteredAndPaginatedPolicies.length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <h4>Policies ({filteredAndPaginatedPolicies.length} on this page)</h4>
                <button onClick={handleSelectAllPolicies} className={styles.selectAllButton}>
                  {allPoliciesSelected ? 'Deselect All' : 'Select All'}
                </button>
              </div>
              {filteredAndPaginatedPolicies.map((policy) => (
                <div
                  key={policy.id}
                  className={`${styles.policyItem} ${isPolicySelected(policy.id, policy.type) ? styles.selected : ''}`}
                  onClick={() => handlePolicyToggle(policy.id, policy.type)}
                >
                  <input
                    type="checkbox"
                    checked={isPolicySelected(policy.id, policy.type)}
                    onChange={() => handlePolicyToggle(policy.id, policy.type)}
                    className={styles.checkbox}
                    onClick={(e) => e.stopPropagation()}
                  />

                  <div className={styles.policyInfo}>
                    <div className={styles.policyName}>{policy.name}</div>
                    <div className={styles.policyMeta}>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>User:</span>
                        <span>{policy.user_id || 'Unknown'}</span>
                      </div>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>Type:</span>
                        <span>{policy.type === 'training_run' ? 'Training Run' : 'Run-free Policy'}</span>
                      </div>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>Created:</span>
                        <span>{formatDate(policy.created_at)}</span>
                      </div>
                      {policy.tags.length > 0 && (
                        <div className={styles.metaItem}>
                          <span className={styles.metaLabel}>Tags:</span>
                          <span>{policy.tags.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className={styles.selectedCount}>
          <span>
            {selectedTrainingRunIds.length + selectedRunFreePolicyIds.length} selected ({selectedTrainingRunIds.length}{' '}
            training runs, {selectedRunFreePolicyIds.length} run-free policies)
            {totalPages > 0 && (
              <span>
                {' '}
                • Page {currentPage} of {totalPages}
              </span>
            )}
          </span>
          <div className={styles.buttonGroup}>
            {currentPage > 1 && (
              <button onClick={() => setCurrentPage(currentPage - 1)} className={styles.pageButton}>
                Previous
              </button>
            )}
            {currentPage < totalPages && (
              <button onClick={() => setCurrentPage(currentPage + 1)} className={styles.pageButton}>
                Next
              </button>
            )}
            <button
              onClick={handleClearAll}
              disabled={selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0}
              className={styles.clearButton}
            >
              Clear All
            </button>
          </div>
        </div>
      </div>
    )
  }
)
