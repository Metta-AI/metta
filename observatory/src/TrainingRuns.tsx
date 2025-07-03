import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { TrainingRun, Repo } from './repo'
import styles from './TrainingRuns.module.css'
import { SearchAndFiltersSidebar } from './SearchAndFiltersSidebar'
import { TrainingRunRow } from './TrainingRunRow'

interface TrainingRunsProps {
  repo: Repo
}

export function TrainingRuns({ repo }: TrainingRunsProps) {
  const [searchParams, setSearchParams] = useSearchParams()
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [currentUser, setCurrentUser] = useState<string | null>(null)
  const [selectedTagFilters, setSelectedTagFilters] = useState<string[]>([])

  // Initialize tag filters from URL parameters
  useEffect(() => {
    const tagFilters = searchParams.get('tag_filters')
    if (tagFilters) {
      // Parse comma-separated values
      const tags = tagFilters
        .split(',')
        .map((tag) => decodeURIComponent(tag.trim()))
        .filter((tag) => tag.length > 0)
      if (tags.length > 0) {
        setSelectedTagFilters(tags)
      }
    }
  }, [searchParams])

  // Update URL when tag filters change
  useEffect(() => {
    const newSearchParams = new URLSearchParams(searchParams)

    if (selectedTagFilters.length > 0) {
      // Join tags with commas and URL encode each tag
      const tagParam = selectedTagFilters.map((tag) => encodeURIComponent(tag)).join(',')
      newSearchParams.set('tag_filters', tagParam)
    } else {
      newSearchParams.delete('tag_filters')
    }

    setSearchParams(newSearchParams, { replace: true })
  }, [selectedTagFilters, searchParams, setSearchParams])

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const [response, userResponse] = await Promise.all([
          repo.getTrainingRuns(),
          repo.whoami().catch(() => ({ user_email: '' })),
        ])
        setTrainingRuns(response.training_runs)
        setCurrentUser(userResponse.user_email)
        setError(null)
      } catch (err: any) {
        setError(`Failed to load training runs: ${err.message}`)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [repo])

  const canEditRun = (run: TrainingRun) => {
    return Boolean(currentUser && run.user_id === currentUser)
  }

  const handleRunUpdate = (updatedRun: TrainingRun) => {
    setTrainingRuns((prev) => prev.map((run) => (run.id === updatedRun.id ? updatedRun : run)))
  }

  // Get all unique tags from all training runs
  const getAllTags = () => {
    const allTags = new Set<string>()
    trainingRuns.forEach((run) => {
      run.tags.forEach((tag) => allTags.add(tag))
    })
    return Array.from(allTags).sort()
  }

  const filteredTrainingRuns = trainingRuns.filter((run) => {
    const query = searchQuery.toLowerCase()
    const matchesSearch =
      run.name.toLowerCase().includes(query) ||
      run.status.toLowerCase().includes(query) ||
      run.user_id.toLowerCase().includes(query) ||
      (run.description && run.description.toLowerCase().includes(query)) ||
      run.tags.some((tag) => tag.toLowerCase().includes(query))

    // Check if run has ALL selected tag filters
    const matchesTagFilters =
      selectedTagFilters.length === 0 || selectedTagFilters.every((filterTag) => run.tags.includes(filterTag))

    return matchesSearch && matchesTagFilters
  })

  if (loading) {
    return (
      <div className={styles.trainingRunsContainer}>
        <div className={styles.trainingRunsContent}>
          <div className={styles.loadingContainer}>
            <div>Loading training runs...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.trainingRunsContainer}>
        <div className={styles.trainingRunsContent}>
          <div className={styles.errorContainer}>
            <div>{error}</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.trainingRunsContainer}>
      <div className={styles.trainingRunsContent}>
        {/* Search and Filters Sidebar */}
        <SearchAndFiltersSidebar
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          selectedTagFilters={selectedTagFilters}
          onTagFiltersChange={setSelectedTagFilters}
          availableTags={getAllTags()}
        />

        {/* Main Content */}
        <div className={styles.mainContent}>
          <div className={styles.trainingRunsHeader}>
            <h1 className={styles.trainingRunsTitle}>Training Runs</h1>
            <div className={styles.trainingRunsCount}>
              {searchQuery || selectedTagFilters.length > 0
                ? `${filteredTrainingRuns.length} of ${trainingRuns.length} training runs`
                : `${trainingRuns.length} training run${trainingRuns.length !== 1 ? 's' : ''}`}
            </div>
          </div>

          {trainingRuns.length === 0 ? (
            <div className={styles.loadingContainer}>
              <div>No training runs found.</div>
            </div>
          ) : filteredTrainingRuns.length === 0 ? (
            <div className={styles.loadingContainer}>
              <div>No training runs match your search.</div>
            </div>
          ) : (
            <table className={styles.trainingRunsTable}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Description</th>
                  <th>Tags</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Finished</th>
                  <th>User</th>
                </tr>
              </thead>
              <tbody>
                {filteredTrainingRuns.map((run) => (
                  <TrainingRunRow
                    key={run.id}
                    run={run}
                    canEdit={canEditRun(run)}
                    repo={repo}
                    onRunUpdate={handleRunUpdate}
                    onError={setError}
                  />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
