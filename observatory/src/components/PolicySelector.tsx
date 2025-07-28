import React, { useState, useEffect } from 'react'
import type { PoliciesResponse, Repo } from '../repo'
import styles from './PolicySelector.module.css'

interface PolicySelectorProps {
  repo: Repo
  searchText: string
  selectedTrainingRunIds: Array<string>
  selectedRunFreePolicyIds: Array<string>
  onTrainingRunSelectionChange: (selectedIds: Array<string>) => void
  onRunFreePolicySelectionChange: (selectedIds: Array<string>) => void
  currentPage: number
  onPageChange: (page: number) => void
  pageSize?: number
}

export const PolicySelector: React.FC<PolicySelectorProps> = React.memo(
  ({
    repo,
    searchText,
    selectedTrainingRunIds,
    selectedRunFreePolicyIds,
    onTrainingRunSelectionChange,
    onRunFreePolicySelectionChange,
    currentPage,
    onPageChange,
    pageSize = 25,
  }) => {
    const [allTrainingRunsSelected, setAllTrainingRunsSelected] = useState(false)
    const [allRunFreePoliciesSelected, setAllRunFreePoliciesSelected] = useState(false)
    const [policiesData, setPoliciesData] = useState<PoliciesResponse | null>(null)
    const [debouncedSearchText, setDebouncedSearchText] = useState<string>('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Helper functions
    const formatDate = (dateStr: string) => {
      try {
        return new Date(dateStr).toLocaleDateString()
      } catch {
        return dateStr
      }
    }

    // Debounce search text input and reset page when search changes
    useEffect(() => {
      const timeoutId = setTimeout(() => {
        setDebouncedSearchText(searchText)
        // Reset to first page when search text changes
        if (currentPage !== 1) {
          onPageChange(1)
        }
      }, 300)

      return () => clearTimeout(timeoutId)
    }, [searchText, currentPage, onPageChange])

    // Load policies when debounced search text, page, or repo changes
    useEffect(() => {
      const loadPolicies = async () => {
        try {
          setLoading(true)
          setError(null)
          const response = await repo.getPolicies({
            search_text: debouncedSearchText,
            pagination: { page: currentPage, page_size: pageSize },
          })
          setPoliciesData(response)
        } catch (err) {
          setError(`Failed to load policies: ${err instanceof Error ? err.message : 'Unknown error'}`)
          setPoliciesData(null)
        } finally {
          setLoading(false)
        }
      }

      loadPolicies()
    }, [repo, debouncedSearchText, currentPage, pageSize])

    // Get unified policies and separate by type
    const allPolicies = policiesData?.policies || []
    const filteredTrainingRuns = allPolicies.filter((p) => p.type === 'training_run')
    const filteredRunFreePolicies = allPolicies.filter((p) => p.type === 'policy')

    // Update selection states when selection changes
    useEffect(() => {
      setAllTrainingRunsSelected(
        filteredTrainingRuns.length > 0 && filteredTrainingRuns.every((run) => selectedTrainingRunIds.includes(run.id))
      )
      setAllRunFreePoliciesSelected(
        filteredRunFreePolicies.length > 0 &&
          filteredRunFreePolicies.every((policy) => selectedRunFreePolicyIds.includes(policy.id))
      )
    }, [selectedTrainingRunIds, selectedRunFreePolicyIds, filteredTrainingRuns, filteredRunFreePolicies])

    const handleTrainingRunToggle = (runId: string) => {
      const newSelection = selectedTrainingRunIds.includes(runId)
        ? selectedTrainingRunIds.filter((id) => id !== runId)
        : [...selectedTrainingRunIds, runId]
      onTrainingRunSelectionChange(newSelection)
    }

    const handleRunFreePolicyToggle = (policyId: string) => {
      const newSelection = selectedRunFreePolicyIds.includes(policyId)
        ? selectedRunFreePolicyIds.filter((id) => id !== policyId)
        : [...selectedRunFreePolicyIds, policyId]
      onRunFreePolicySelectionChange(newSelection)
    }

    const handleSelectAllTrainingRuns = () => {
      if (allTrainingRunsSelected) {
        // Deselect all visible training runs
        const visibleIds = new Set(filteredTrainingRuns.map((r) => r.id))
        const newSelection = selectedTrainingRunIds.filter((id) => !visibleIds.has(id))
        onTrainingRunSelectionChange(newSelection)
      } else {
        // Select all visible training runs
        const visibleIds = filteredTrainingRuns.map((r) => r.id)
        const newSelection = [...new Set([...selectedTrainingRunIds, ...visibleIds])]
        onTrainingRunSelectionChange(newSelection)
      }
    }

    const handleSelectAllRunFreePolicies = () => {
      if (allRunFreePoliciesSelected) {
        // Deselect all visible run-free policies
        const visibleIds = new Set(filteredRunFreePolicies.map((p) => p.id))
        const newSelection = selectedRunFreePolicyIds.filter((id) => !visibleIds.has(id))
        onRunFreePolicySelectionChange(newSelection)
      } else {
        // Select all visible run-free policies
        const visibleIds = filteredRunFreePolicies.map((p) => p.id)
        const newSelection = [...new Set([...selectedRunFreePolicyIds, ...visibleIds])]
        onRunFreePolicySelectionChange(newSelection)
      }
    }

    const handleClearAll = () => {
      onTrainingRunSelectionChange([])
      onRunFreePolicySelectionChange([])
    }

    return (
      <div className={styles.container}>
        {error && (
          <div className={styles.errorContainer}>
            <div className={styles.errorMessage}>{error}</div>
          </div>
        )}

        <div className={styles.policiesContainer}>
          {loading && (
            <div className={styles.loadingContainer}>
              <span className={styles.loadingSpinner} />
              Loading policies...
            </div>
          )}

          {!loading && filteredTrainingRuns.length === 0 && filteredRunFreePolicies.length === 0 && (
            <div className={styles.emptyMessage}>
              {debouncedSearchText
                ? 'No training runs or policies match your search.'
                : 'No training runs or policies available.'}
            </div>
          )}

          {!loading && filteredTrainingRuns.length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <h4>Training Runs ({filteredTrainingRuns.length})</h4>
                <button onClick={handleSelectAllTrainingRuns} className={styles.selectAllButton}>
                  {allTrainingRunsSelected ? 'Deselect All' : 'Select All'}
                </button>
              </div>
              {filteredTrainingRuns.map((run) => (
                <div
                  key={run.id}
                  className={`${styles.policyItem} ${selectedTrainingRunIds.includes(run.id) ? styles.selected : ''}`}
                  onClick={() => handleTrainingRunToggle(run.id)}
                >
                  <input
                    type="checkbox"
                    checked={selectedTrainingRunIds.includes(run.id)}
                    onChange={() => handleTrainingRunToggle(run.id)}
                    className={styles.checkbox}
                    onClick={(e) => e.stopPropagation()}
                  />

                  <div className={styles.policyInfo}>
                    <div className={styles.policyName}>{run.name}</div>
                    <div className={styles.policyMeta}>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>User:</span>
                        <span>{run.user_id || 'Unknown'}</span>
                      </div>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>Type:</span>
                        <span>Training Run</span>
                      </div>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>Created:</span>
                        <span>{formatDate(run.created_at)}</span>
                      </div>
                      {run.tags.length > 0 && (
                        <div className={styles.metaItem}>
                          <span className={styles.metaLabel}>Tags:</span>
                          <span>{run.tags.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {!loading && filteredRunFreePolicies.length > 0 && (
            <div className={styles.section}>
              <div className={styles.sectionHeader}>
                <h4>Run-free Policies ({filteredRunFreePolicies.length})</h4>
                <button onClick={handleSelectAllRunFreePolicies} className={styles.selectAllButton}>
                  {allRunFreePoliciesSelected ? 'Deselect All' : 'Select All'}
                </button>
              </div>
              {filteredRunFreePolicies.map((policy) => (
                <div
                  key={policy.id}
                  className={`${styles.policyItem} ${
                    selectedRunFreePolicyIds.includes(policy.id) ? styles.selected : ''
                  }`}
                  onClick={() => handleRunFreePolicyToggle(policy.id)}
                >
                  <input
                    type="checkbox"
                    checked={selectedRunFreePolicyIds.includes(policy.id)}
                    onChange={() => handleRunFreePolicyToggle(policy.id)}
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
                        <span>Run-free Policy</span>
                      </div>
                      <div className={styles.metaItem}>
                        <span className={styles.metaLabel}>Created:</span>
                        <span>{formatDate(policy.created_at)}</span>
                      </div>
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
            {policiesData && (
              <span>
                {' '}
                • Page {policiesData.page} of {Math.ceil(policiesData.total_count / policiesData.page_size)}
              </span>
            )}
          </span>
          <div className={styles.buttonGroup}>
            {policiesData && currentPage > 1 && (
              <button onClick={() => onPageChange(currentPage - 1)} className={styles.pageButton}>
                Previous
              </button>
            )}
            {policiesData && currentPage < Math.ceil(policiesData.total_count / policiesData.page_size) && (
              <button onClick={() => onPageChange(currentPage + 1)} className={styles.pageButton}>
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
