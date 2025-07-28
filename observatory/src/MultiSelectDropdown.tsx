import { useEffect, useState } from 'react'

interface MultiSelectOption {
  value: string
  label: string
  metadata?: Record<string, unknown>
}

interface MultiSelectDropdownProps {
  options: Array<MultiSelectOption>
  selectedValues: Set<string>
  onSelectionChange: (selectedValues: Set<string>) => void
  placeholder?: string
  searchPlaceholder?: string
  maxHeight?: number
  width?: string
}

// CSS for multi-select dropdown
const MULTI_SELECT_CSS = `
.multi-select-dropdown-container {
  position: relative;
  width: 100%;
}

.multi-select-dropdown {
  position: relative;
  width: 100%;
}

.multi-select-dropdown-trigger {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  color: #333;
  transition: all 0.2s ease;
}

.multi-select-dropdown-trigger:hover {
  border-color: #007bff;
}

.multi-select-dropdown-trigger.open {
  border-color: #007bff;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}

.multi-select-dropdown-content {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: #fff;
  border: 1px solid #007bff;
  border-top: none;
  border-radius: 0 0 4px 4px;
  max-height: 300px;
  overflow: hidden;
  z-index: 1000;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.multi-select-search {
  padding: 12px 16px;
  border-bottom: 1px solid #eee;
}

.multi-select-search-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  outline: none;
}

.multi-select-search-input:focus {
  border-color: #007bff;
}

.multi-select-list {
  max-height: 200px;
  overflow-y: auto;
}

.multi-select-item {
  display: flex;
  align-items: center;
  padding: 10px 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  border-bottom: 1px solid #f0f0f0;
}

.multi-select-item:last-child {
  border-bottom: none;
}

.multi-select-item:hover {
  background: #f8f9fa;
}

.multi-select-item.selected {
  background: #e3f2fd;
}

.multi-select-checkbox {
  margin-right: 12px;
  cursor: pointer;
}

.multi-select-label {
  flex: 1;
  font-size: 14px;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.multi-select-metadata {
  font-size: 12px;
  color: #666;
  margin-left: 10px;
  font-weight: 500;
}

.multi-select-dropdown-arrow {
  transition: transform 0.2s ease;
}

.multi-select-dropdown-arrow.open {
  transform: rotate(180deg);
}
`

export function MultiSelectDropdown({
  options,
  selectedValues,
  onSelectionChange,
  placeholder = 'Select options',
  searchPlaceholder = 'Search...',
  maxHeight = 300,
  width = '100%',
}: MultiSelectDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')

  // Filter options based on search term
  const filteredOptions = options.filter((option) => option.label.toLowerCase().includes(searchTerm.toLowerCase()))

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element
      if (!target.closest('.multi-select-dropdown')) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  const toggleSelection = (value: string) => {
    const newSelectedValues = new Set(selectedValues)
    if (newSelectedValues.has(value)) {
      newSelectedValues.delete(value)
    } else {
      newSelectedValues.add(value)
    }
    onSelectionChange(newSelectedValues)
  }

  const getDisplayText = () => {
    if (selectedValues.size === 0) {
      return placeholder
    }
    if (selectedValues.size === 1) {
      const option = options.find((opt) => opt.value === Array.from(selectedValues)[0])
      return option?.label || placeholder
    }
    return `${selectedValues.size} items selected`
  }

  return (
    <>
      <style>{MULTI_SELECT_CSS}</style>
      <div className="multi-select-dropdown-container" style={{ width }}>
        <div className="multi-select-dropdown">
          <div 
            className={`multi-select-dropdown-trigger ${isOpen ? 'open' : ''}`} 
            onClick={() => setIsOpen(!isOpen)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                setIsOpen(!isOpen)
              }
            }}
            role="button"
            tabIndex={0}
          >
            {getDisplayText()}
            <span className={`multi-select-dropdown-arrow ${isOpen ? 'open' : ''}`}>▼</span>
          </div>
          {isOpen && (
            <div className="multi-select-dropdown-content" style={{ maxHeight }}>
              <div className="multi-select-search">
                <input
                  type="text"
                  className="multi-select-search-input"
                  placeholder={searchPlaceholder}
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              <div className="multi-select-list">
                {filteredOptions.map((option) => (
                  <div
                    key={option.value}
                    className={`multi-select-item ${selectedValues.has(option.value) ? 'selected' : ''}`}
                    onClick={() => toggleSelection(option.value)}
                  >
                    <input
                      type="checkbox"
                      className="multi-select-checkbox"
                      checked={selectedValues.has(option.value)}
                      readOnly
                    />
                    <span className="multi-select-label">{option.label}</span>
                    {option.metadata && (
                      <span className="multi-select-metadata">
                        {Object.entries(option.metadata)
                          .map(([_, value]) => (typeof value === 'number' ? value.toFixed(3) : value))
                          .join(' ')}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
