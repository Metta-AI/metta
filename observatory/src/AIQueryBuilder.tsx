import { useState } from 'react'
import type { Repo } from './repo'

interface Props {
  repo: Repo
  onQueryGenerated: (query: string) => void
}

const AI_QUERY_CSS = `
  .ai-query-builder {
    background-color: #f0f9ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .ai-query-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .ai-query-title {
    font-size: 14px;
    font-weight: 600;
    color: #1e40af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0;
  }

  .ai-query-input {
    width: 100%;
    padding: 10px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    font-size: 13px;
    background-color: white;
    margin-bottom: 12px;
    resize: vertical;
    min-height: 60px;
    font-family: inherit;
    line-height: 1.5;
    box-sizing: border-box;
  }

  .ai-query-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .error-message {
    color: #dc2626;
    font-size: 12px;
    margin-top: 8px;
  }
`

export function AIQueryBuilder({ repo, onQueryGenerated }: Props) {
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generateQuery = async () => {
    const trimmedDescription = description.trim()
    if (!trimmedDescription) {
      setError('Please describe your query')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const { query } = await repo.generateAIQuery(trimmedDescription)
      onQueryGenerated(query)
      setDescription('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate query')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      generateQuery()
    }
  }

  return (
    <>
      <style>{AI_QUERY_CSS}</style>
      <div className="ai-query-builder">
        <div className="ai-query-header">
          <h3 className="ai-query-title">Describe your query</h3>
        </div>

        <textarea
          className="ai-query-input"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g., Show me the top 10 training runs by average reward"
          disabled={loading}
        />
        <div className="ai-query-buttons">
          <button className="btn btn-primary" onClick={generateQuery} disabled={loading || !description.trim()}>
            {loading ? 'Generating...' : 'Generate Query'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}
      </div>
    </>
  )
}
