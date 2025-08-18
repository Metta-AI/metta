import React from 'react'
import styles from './PolicySelector.module.css'

interface SearchInputProps {
  searchText: string
  onSearchChange: (searchText: string) => void
  disabled?: boolean
}

export const SearchInput: React.FC<SearchInputProps> = React.memo(
  ({ searchText, onSearchChange, disabled = false }) => {
    return (
      <div className={styles.searchContainer}>
        <input
          type="text"
          placeholder="Search by training run name, policy name, user, or date..."
          value={searchText}
          onChange={(e) => onSearchChange(e.target.value)}
          className={styles.searchBox}
          disabled={disabled}
        />
      </div>
    )
  }
)
