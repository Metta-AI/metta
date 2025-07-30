import { useEffect, useState } from 'react'
import { Episode, Repo } from './repo'

interface EpisodesProps {
  repo: Repo
}

export function Episodes({ repo }: EpisodesProps) {
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filterQuery, setFilterQuery] = useState('')
  const [appliedFilterQuery, setAppliedFilterQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [totalCount, setTotalCount] = useState(0)
  const [selectedEpisodes, setSelectedEpisodes] = useState<Set<string>>(new Set())
  const [selectAllChecked, setSelectAllChecked] = useState(false)
  const [tagInput, setTagInput] = useState('')
  const [tagAction, setTagAction] = useState<'add' | 'remove'>('add')
  const [allTags, setAllTags] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1)

  const pageSize = 50

  useEffect(() => {
    const loadEpisodes = async () => {
      try {
        setLoading(true)
        const response = await repo.filterEpisodes(currentPage, pageSize, appliedFilterQuery)
        setEpisodes(response.episodes)
        setTotalPages(response.total_pages)
        setTotalCount(response.total_count)
        setError(null)
      } catch (err: any) {
        setError(`Failed to load episodes: ${err.message}`)
      } finally {
        setLoading(false)
      }
    }

    loadEpisodes()
  }, [repo, currentPage, appliedFilterQuery])

  useEffect(() => {
    const loadAllTags = async () => {
      try {
        const response = await repo.getAllEpisodeTags()
        setAllTags(response.tags)
      } catch (err: any) {
        // Don't show error for tags loading failure, just log it
        console.warn('Failed to load existing tags:', err.message)
      }
    }

    loadAllTags()
  }, [repo])

  const handleFilterSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setCurrentPage(1)
    setAppliedFilterQuery(filterQuery)
    // Clear selections when filter changes
    setSelectedEpisodes(new Set())
    setSelectAllChecked(false)
  }

  const handleSelectAll = () => {
    setSelectAllChecked(!selectAllChecked)
    if (selectAllChecked) {
      // If currently checked, clear all selections
      setSelectedEpisodes(new Set())
    } else {
      // If currently unchecked, select all episodes on current page
      setSelectedEpisodes(new Set(episodes.map((ep) => ep.id)))
    }
  }

  const handleEpisodeSelect = (episodeId: string, checked: boolean) => {
    const newSelected = new Set(selectedEpisodes)
    if (checked) {
      newSelected.add(episodeId)
    } else {
      newSelected.delete(episodeId)
      // If any individual episode is unselected, uncheck "select all"
      setSelectAllChecked(false)
    }
    setSelectedEpisodes(newSelected)
  }

  const handleTagInputChange = (value: string) => {
    setTagInput(value)

    if (value.trim() === '') {
      setShowSuggestions(false)
      setFilteredSuggestions([])
      setSelectedSuggestionIndex(-1)
    } else {
      const filtered = allTags.filter((tag) => tag.toLowerCase().includes(value.toLowerCase()) && tag !== value)
      setFilteredSuggestions(filtered)
      setShowSuggestions(filtered.length > 0)
      setSelectedSuggestionIndex(-1)
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setTagInput(suggestion)
    setShowSuggestions(false)
    setFilteredSuggestions([])
    setSelectedSuggestionIndex(-1)
  }

  const handleTagInputBlur = () => {
    // Add a small delay to allow click events on suggestions to fire
    setTimeout(() => {
      setShowSuggestions(false)
    }, 150)
  }

  const handleTagInputFocus = () => {
    if (tagInput.trim() !== '' && filteredSuggestions.length > 0) {
      setShowSuggestions(true)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || filteredSuggestions.length === 0) return

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedSuggestionIndex((prev) => (prev < filteredSuggestions.length - 1 ? prev + 1 : prev))
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedSuggestionIndex((prev) => (prev > 0 ? prev - 1 : prev))
        break
      case 'Enter':
        e.preventDefault()
        if (selectedSuggestionIndex >= 0) {
          handleSuggestionClick(filteredSuggestions[selectedSuggestionIndex])
        }
        break
      case 'Escape':
        setShowSuggestions(false)
        setSelectedSuggestionIndex(-1)
        break
    }
  }

  const handleTagAction = async () => {
    if (!tagInput.trim() || (selectedEpisodes.size === 0 && !selectAllChecked)) return

    try {
      setLoading(true)

      if (selectAllChecked) {
        // Use filter-based tagging for all episodes in the current filter
        if (tagAction === 'add') {
          await repo.addEpisodeTagsByFilter(appliedFilterQuery, tagInput.trim())
        } else {
          await repo.removeEpisodeTagsByFilter(appliedFilterQuery, tagInput.trim())
        }
      } else {
        // Use individual episode tagging for selected episodes
        const episodeIds = Array.from(selectedEpisodes)
        if (tagAction === 'add') {
          await repo.addEpisodeTags(episodeIds, tagInput.trim())
        } else {
          await repo.removeEpisodeTags(episodeIds, tagInput.trim())
        }
      }

      // Refresh the episodes list
      const response = await repo.filterEpisodes(currentPage, pageSize, appliedFilterQuery)
      setEpisodes(response.episodes)
      setSelectedEpisodes(new Set())
      setSelectAllChecked(false)
      setTagInput('')
      setError(null)

      // Reload tags in case we added a new one
      const tagsResponse = await repo.getAllEpisodeTags()
      setAllTags(tagsResponse.tags)
    } catch (err: any) {
      setError(`Failed to ${tagAction} tags: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  if (loading && episodes.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <p>Loading episodes...</p>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>Episodes</h1>

      {/* Error Display */}
      {error && (
        <div
          style={{
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '5px',
            color: '#721c24',
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Filter Section */}
      <div
        style={{
          marginBottom: '20px',
          padding: '20px',
          backgroundColor: '#f8f9fa',
          borderRadius: '5px',
        }}
      >
        <form onSubmit={handleFilterSubmit}>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label htmlFor="filter-query">Filter:</label>
            <input
              id="filter-query"
              type="text"
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              placeholder="e.g., policy_name='test' AND training_run_name='run1' AND eval_category='navigation'"
              style={{
                flex: 1,
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '14px',
              }}
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '8px 16px',
                backgroundColor: loading ? '#6c757d' : '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? 'Applying...' : 'Apply Filter'}
            </button>
          </div>
        </form>
        {appliedFilterQuery && (
          <div style={{ marginTop: '10px', fontSize: '14px', color: '#6c757d' }}>
            <strong>Applied filter:</strong> {appliedFilterQuery}
          </div>
        )}
        <div style={{ marginTop: '10px', fontSize: '12px', color: '#6c757d' }}>
          <strong>Available filter fields:</strong> policy_name, policy_description, training_run_name,
          training_run_status, training_run_user_id, created_at, eval_name, eval_category, env_name, simulation_suite
        </div>
      </div>

      {/* Tag Management Section */}
      <div
        style={{
          marginBottom: '20px',
          padding: '20px',
          backgroundColor: '#f8f9fa',
          borderRadius: '5px',
        }}
      >
        <h3>Tag Management</h3>
        <div
          style={{
            display: 'flex',
            gap: '10px',
            alignItems: 'center',
            marginBottom: '10px',
          }}
        >
          <div style={{ position: 'relative', flex: '0 0 200px' }}>
            <input
              type="text"
              value={tagInput}
              onChange={(e) => handleTagInputChange(e.target.value)}
              onBlur={handleTagInputBlur}
              onFocus={handleTagInputFocus}
              onKeyDown={handleKeyDown}
              placeholder="Enter tag name"
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '14px',
              }}
            />
            {showSuggestions && filteredSuggestions.length > 0 && (
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  backgroundColor: 'white',
                  border: '1px solid #ddd',
                  borderTop: 'none',
                  borderRadius: '0 0 4px 4px',
                  maxHeight: '200px',
                  overflowY: 'auto',
                  zIndex: 1000,
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                }}
              >
                {filteredSuggestions.map((suggestion, index) => (
                  <div
                    key={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    style={{
                      padding: '8px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      borderBottom: '1px solid #eee',
                      backgroundColor: index === selectedSuggestionIndex ? '#e3f2fd' : 'white',
                    }}
                    onMouseEnter={(e) => {
                      if (index !== selectedSuggestionIndex) {
                        e.currentTarget.style.backgroundColor = '#f8f9fa'
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = index === selectedSuggestionIndex ? '#e3f2fd' : 'white'
                    }}
                  >
                    {suggestion}
                  </div>
                ))}
              </div>
            )}
          </div>
          <select
            value={tagAction}
            onChange={(e) => setTagAction(e.target.value as 'add' | 'remove')}
            style={{
              padding: '8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
            }}
          >
            <option value="add">Add Tag</option>
            <option value="remove">Remove Tag</option>
          </select>
          <button
            onClick={handleTagAction}
            disabled={!tagInput.trim() || (selectedEpisodes.size === 0 && !selectAllChecked)}
            style={{
              padding: '8px 16px',
              backgroundColor: tagAction === 'add' ? '#28a745' : '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: selectedEpisodes.size === 0 && !selectAllChecked ? 'not-allowed' : 'pointer',
              opacity: selectedEpisodes.size === 0 && !selectAllChecked ? 0.6 : 1,
            }}
          >
            {tagAction === 'add' ? 'Add' : 'Remove'} Tag to{' '}
            {selectAllChecked ? `all ${totalCount} filtered episodes` : `${selectedEpisodes.size} episode(s)`}
          </button>
        </div>
      </div>

      {/* Episodes Table */}
      <div style={{ marginBottom: '20px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '10px',
          }}
        >
          <p>
            Showing {episodes.length} episodes (Page {currentPage} of {totalPages}, Total: {totalCount})
            {(selectedEpisodes.size > 0 || selectAllChecked) && (
              <span
                style={{
                  marginLeft: '10px',
                  color: '#007bff',
                  fontWeight: 'bold',
                }}
              >
                {selectAllChecked ? `All ${totalCount} episodes selected` : `${selectedEpisodes.size} selected`}
              </span>
            )}
          </p>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              cursor: 'pointer',
            }}
          >
            <input type="checkbox" checked={selectAllChecked} onChange={handleSelectAll} style={{ margin: 0 }} />
            Select All ({totalCount} episodes)
          </label>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '14px',
            }}
          >
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa' }}>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  Select
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  ID
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  created_at
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  policy_name
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  training_run_name
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  eval_category
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  env_name
                </th>
                <th
                  style={{
                    padding: '12px',
                    textAlign: 'left',
                    border: '1px solid #ddd',
                  }}
                >
                  Tags
                </th>
              </tr>
            </thead>
            <tbody>
              {episodes.map((episode) => (
                <tr key={episode.id}>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>
                    <input
                      type="checkbox"
                      checked={selectedEpisodes.has(episode.id)}
                      onChange={(e) => handleEpisodeSelect(episode.id, e.target.checked)}
                    />
                  </td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>
                    <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>{episode.id.substring(0, 8)}...</span>
                  </td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{formatDateTime(episode.created_at)}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{episode.policy_name || 'N/A'}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{episode.training_run_name || 'N/A'}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{episode.eval_category || 'N/A'}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>{episode.env_name || 'N/A'}</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd' }}>
                    {episode.tags.length > 0 ? (
                      <div
                        style={{
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: '4px',
                        }}
                      >
                        {episode.tags.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              padding: '2px 8px',
                              backgroundColor: '#007bff',
                              color: 'white',
                              fontSize: '12px',
                              borderRadius: '12px',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span style={{ color: '#6c757d' }}>No tags</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '10px',
          alignItems: 'center',
        }}
      >
        <button
          onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
          disabled={currentPage === 1}
          style={{
            padding: '8px 16px',
            backgroundColor: currentPage === 1 ? '#6c757d' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
          }}
        >
          Previous
        </button>

        <span>
          Page {currentPage} of {totalPages}
        </span>

        <button
          onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
          disabled={currentPage === totalPages}
          style={{
            padding: '8px 16px',
            backgroundColor: currentPage === totalPages ? '#6c757d' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
          }}
        >
          Next
        </button>
      </div>
    </div>
  )
}
