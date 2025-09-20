import { useState } from 'react'

interface TypeaheadInputProps {
  value: string
  onChange: (value: string) => void
  placeholder: string
  suggestions: string[]
  maxSuggestions?: number
  filterType?: 'prefix' | 'substring'
}

function TypeaheadInput({
  value,
  onChange,
  placeholder,
  suggestions,
  maxSuggestions = 10,
  filterType = 'substring',
}: TypeaheadInputProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])

  const handleInputChange = (inputValue: string) => {
    onChange(inputValue)
    if (inputValue.trim()) {
      const filtered = suggestions.filter((suggestion) => {
        const lowerSuggestion = suggestion.toLowerCase()
        const lowerInput = inputValue.toLowerCase()
        return filterType === 'prefix' ? lowerSuggestion.startsWith(lowerInput) : lowerSuggestion.includes(lowerInput)
      })
      setFilteredSuggestions(filtered.slice(0, maxSuggestions))
      setShowSuggestions(filtered.length > 0)
    } else {
      setShowSuggestions(false)
    }
  }

  return (
    <div style={{ position: 'relative' }}>
      <input
        type="text"
        value={value}
        onChange={(e) => handleInputChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%',
          padding: '10px 12px',
          borderRadius: '6px',
          border: '1px solid #d1d5db',
          fontSize: '14px',
          backgroundColor: '#fff',
          transition: 'border-color 0.2s',
          outline: 'none',
        }}
        onFocus={(e) => {
          e.target.style.borderColor = '#007bff'
          if (value.trim() && filteredSuggestions.length > 0) {
            setShowSuggestions(true)
          }
        }}
        onBlur={(e) => {
          e.target.style.borderColor = '#d1d5db'
          // Delay to allow clicking on suggestions
          setTimeout(() => setShowSuggestions(false), 200)
        }}
      />
      {showSuggestions && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            marginTop: '4px',
            backgroundColor: 'white',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            maxHeight: '200px',
            overflowY: 'auto',
            zIndex: 1000,
          }}
        >
          {filteredSuggestions.map((suggestion) => (
            <div
              key={suggestion}
              onClick={() => {
                onChange(suggestion)
                setShowSuggestions(false)
              }}
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                fontSize: '14px',
                borderBottom: '1px solid #f0f0f0',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f8f9fa')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'white')}
            >
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default TypeaheadInput
