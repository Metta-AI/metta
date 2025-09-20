import React from 'react'
import { AttributesRendererProps } from '../types/evalTasks'

export const AttributesRenderer: React.FC<AttributesRendererProps> = ({ attributes }) => {
  if (!attributes || Object.keys(attributes).length === 0) {
    return <div style={{ padding: '10px', color: '#6c757d' }}>No attributes</div>
  }

  const formatValue = (value: any): React.ReactNode => {
    if (typeof value === 'string') {
      // Split by newlines and render each line separately
      const lines = value.split('\\n')
      if (lines.length > 1) {
        return (
          <>
            {lines.map((line, i) => (
              <React.Fragment key={i}>
                {i > 0 && <br />}
                {line}
              </React.Fragment>
            ))}
          </>
        )
      }
      return value
    }
    return JSON.stringify(value, null, 2)
  }

  const renderObject = (obj: Record<string, any>, indent: number = 0): React.ReactNode => {
    // Filter out empty/falsy values
    const entries = Object.entries(obj).filter(([_, value]) => {
      if (value === null || value === undefined || value === '' || value === false) return false
      if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0) return false
      if (Array.isArray(value) && value.length === 0) return false
      return true
    })

    return (
      <div style={{ marginLeft: indent > 0 ? '20px' : 0 }}>
        {entries.map(([key, value], i) => (
          <div key={key} style={{ marginBottom: i < entries.length - 1 ? '8px' : 0 }}>
            <span style={{ color: '#0066cc', fontWeight: 500 }}>{key}:</span>{' '}
            {typeof value === 'object' && value !== null && !Array.isArray(value) ? (
              renderObject(value, indent + 1)
            ) : (
              <span style={{ color: '#333' }}>{formatValue(value)}</span>
            )}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div
      style={{
        padding: '15px',
        backgroundColor: '#f8f9fa',
        borderTop: '1px solid #dee2e6',
      }}
    >
      <h4
        style={{
          marginTop: 0,
          marginBottom: '10px',
          fontSize: '14px',
          fontWeight: 600,
        }}
      >
        Attributes
      </h4>
      <div
        style={{
          fontSize: '12px',
          lineHeight: 1.6,
          fontFamily: 'Monaco, Consolas, "Courier New", monospace',
        }}
      >
        {renderObject(attributes)}
      </div>
    </div>
  )
}
