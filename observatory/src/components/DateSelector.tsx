import React from 'react'

interface DateSelectorProps {
  value: string // ISO date string
  onChange: (date: string) => void
  label?: string
  disabled?: boolean
}

export function DateSelector({ value, onChange, label = 'Start Date', disabled = false }: DateSelectorProps) {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value)
  }

  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#333' }}>{label}</label>
      <input
        type="date"
        value={value}
        onChange={handleChange}
        disabled={disabled}
        style={{
          padding: '8px 12px',
          border: '1px solid #ddd',
          borderRadius: '4px',
          fontSize: '14px',
          width: '100%',
          maxWidth: '200px',
          backgroundColor: disabled ? '#f5f5f5' : 'white',
          color: disabled ? '#666' : '#333',
          cursor: disabled ? 'not-allowed' : 'text',
        }}
      />
    </div>
  )
}
