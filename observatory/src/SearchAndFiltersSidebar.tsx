import { useState } from 'react'
import styles from './TrainingRuns.module.css'

interface SearchAndFiltersSidebarProps {
  searchQuery: string
  onSearchQueryChange: (query: string) => void
  selectedTagFilters: string[]
  onTagFiltersChange: (filters: string[]) => void
  availableTags: string[]
}

export function SearchAndFiltersSidebar({
  searchQuery,
  onSearchQueryChange,
  selectedTagFilters,
  onTagFiltersChange,
  availableTags,
}: SearchAndFiltersSidebarProps) {
  const [tagFilterInput, setTagFilterInput] = useState('')
  const [showTagDropdown, setShowTagDropdown] = useState(false)

  // Filter available tags based on input and exclude already selected ones
  const getFilteredAvailableTags = () => {
    const availableTagsFiltered = availableTags.filter((tag) => !selectedTagFilters.includes(tag))

    if (!tagFilterInput.trim()) {
      return availableTagsFiltered
    }

    return availableTagsFiltered.filter((tag) => tag.toLowerCase().includes(tagFilterInput.toLowerCase()))
  }

  const handleTagFilterInputChange = (value: string) => {
    setTagFilterInput(value)
    setShowTagDropdown(true)
  }

  const handleTagFilterInputFocus = () => {
    setShowTagDropdown(true)
  }

  const handleTagFilterInputBlur = () => {
    // Delay hiding dropdown to allow clicking on items
    setTimeout(() => setShowTagDropdown(false), 150)
  }

  const handleSelectTagFilter = (tag: string) => {
    if (!selectedTagFilters.includes(tag)) {
      onTagFiltersChange([...selectedTagFilters, tag])
    }
    setTagFilterInput('')
    setShowTagDropdown(false)
  }

  const handleRemoveTagFilter = (tagToRemove: string) => {
    onTagFiltersChange(selectedTagFilters.filter((tag) => tag !== tagToRemove))
  }

  const handleClearAllFilters = () => {
    onTagFiltersChange([])
  }

  return (
    <div className={styles.sidebar}>
      <h2 className={styles.sidebarTitle}>Search & Filters</h2>

      <div className={styles.sidebarSection}>
        <label className={styles.sidebarLabel}>Search</label>
        <input
          type="text"
          placeholder="Search training runs..."
          value={searchQuery}
          onChange={(e) => onSearchQueryChange(e.target.value)}
          className={styles.searchBox}
        />
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.tagFilterSection}>
          <div className={styles.tagFilterLabel}>Filter by Tags</div>
          <div className={styles.tagFilterInputContainer}>
            <input
              type="text"
              placeholder="Search tags to filter by..."
              value={tagFilterInput}
              onChange={(e) => handleTagFilterInputChange(e.target.value)}
              onFocus={handleTagFilterInputFocus}
              onBlur={handleTagFilterInputBlur}
              className={styles.tagFilterDropdownInput}
            />
            {showTagDropdown && (
              <div className={styles.tagDropdown}>
                {getFilteredAvailableTags().length > 0 ? (
                  getFilteredAvailableTags().map((tag) => (
                    <div key={tag} className={styles.tagDropdownItem} onMouseDown={() => handleSelectTagFilter(tag)}>
                      {tag}
                    </div>
                  ))
                ) : (
                  <div className={styles.tagDropdownEmpty}>
                    {tagFilterInput ? 'No matching tags found' : 'No more tags available'}
                  </div>
                )}
              </div>
            )}
          </div>
          {selectedTagFilters.length > 0 && (
            <div className={styles.selectedFiltersSection}>
              <div className={styles.selectedFiltersLabel}>Active Filters</div>
              <div className={styles.selectedFilters}>
                {selectedTagFilters.map((tag) => (
                  <span key={tag} className={styles.filterTag}>
                    {tag}
                    <button
                      onClick={() => handleRemoveTagFilter(tag)}
                      className={styles.filterTagRemove}
                      title="Remove filter"
                    >
                      ×
                    </button>
                  </span>
                ))}
                <button onClick={handleClearAllFilters} className={styles.clearAllFiltersBtn}>
                  Clear All
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
