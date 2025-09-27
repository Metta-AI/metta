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
      <div className={styles.searchContainer} style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        <span
          style={{
            position: 'absolute',
            left: 12,
            top: '50%',
            transform: 'translateY(-50%)',
            pointerEvents: 'none',
            color: '#b0b0b0',
            display: 'flex',
            alignItems: 'center',
            fontSize: 18,
            zIndex: 2,
          }}
          aria-hidden="true"
        >
          {/* Simple magnifier SVG */}
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
            <circle cx="9" cy="9" r="7" stroke="#b0b0b0" strokeWidth="2" />
            <line x1="14.2" y1="14.2" x2="18" y2="18" stroke="#b0b0b0" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </span>
        <input
          type="text"
          placeholder="Search by training run name, policy name, user, or date..."
          value={searchText}
          onChange={(e) => onSearchChange(e.target.value)}
          className={styles.searchBox}
          disabled={disabled}
          style={{
            paddingLeft: 38, // make room for icon
          }}
        />
      </div>
    )
  }
)
