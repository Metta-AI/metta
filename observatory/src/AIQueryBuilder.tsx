import { useState, useEffect } from 'react'
import { Repo, TableSchema } from './repo'

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
  }

  .ai-query-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .ai-query-buttons {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .ai-status {
    font-size: 12px;
    color: #64748b;
    font-style: italic;
  }

  .api-key-section {
    background-color: #fef3c7;
    border: 1px solid #fcd34d;
    border-radius: 4px;
    padding: 12px;
    margin-top: 8px;
  }

  .api-key-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: #92400e;
    margin-bottom: 6px;
  }

  .api-key-input {
    width: 100%;
    padding: 8px;
    border: 1px solid #fcd34d;
    border-radius: 4px;
    font-size: 12px;
    font-family: monospace;
    margin-bottom: 8px;
  }

  .api-key-buttons {
    display: flex;
    gap: 8px;
  }

  .btn-sm {
    padding: 4px 12px;
    font-size: 12px;
  }

  .btn-secondary {
    background-color: #f3f4f6;
    color: #374151;
    border: 1px solid #d1d5db;
  }

  .btn-secondary:hover {
    background-color: #e5e7eb;
  }

  .error-message {
    color: #dc2626;
    font-size: 12px;
    margin-top: 8px;
  }
`

export function AIQueryBuilder({ repo, onQueryGenerated }: Props) {
  const [description, setDescription] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showApiKeyInput, setShowApiKeyInput] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tableSchemas, setTableSchemas] = useState<TableSchema[]>([])
  const [schemasLoading, setSchemasLoading] = useState(false)

  useEffect(() => {
    const savedKey = localStorage.getItem('ANTHROPIC_API_KEY')
    if (savedKey) {
      setApiKey(savedKey)
    } else {
      setShowApiKeyInput(true)
    }
  }, [])

  async function loadAllSchemas() {
    if (tableSchemas.length > 0) return

    try {
      setSchemasLoading(true)
      const tables = await repo.listTables()
      const schemas = await Promise.all(tables.map((table) => repo.getTableSchema(table.table_name)))
      setTableSchemas(schemas)
    } catch (err) {
      console.error('Failed to load schemas:', err)
      setError('Failed to load table schemas')
    } finally {
      setSchemasLoading(false)
    }
  }

  function saveApiKey() {
    if (apiKey.trim()) {
      localStorage.setItem('ANTHROPIC_API_KEY', apiKey.trim())
      setShowApiKeyInput(false)
      setError(null)
    }
  }

  function clearApiKey() {
    localStorage.removeItem('ANTHROPIC_API_KEY')
    setApiKey('')
    setShowApiKeyInput(true)
  }

  async function generateQuery() {
    if (!description.trim()) {
      setError('Please describe your query')
      return
    }

    if (!apiKey) {
      setShowApiKeyInput(true)
      setError('Please provide an API key')
      return
    }

    await loadAllSchemas()
    if (schemasLoading) return

    setLoading(true)
    setError(null)

    try {
      const schemaDescription = tableSchemas
        .map((schema) => {
          const columns = schema.columns
            .map(
              (col) =>
                `    ${col.name} ${col.type}${col.nullable ? ' NULL' : ' NOT NULL'}${col.default ? ` DEFAULT ${col.default}` : ''}`
            )
            .join('\n')
          return `Table: ${schema.table_name}\n${columns}`
        })
        .join('\n\n')

      const prompt = `You are a SQL query generator. Given the following database schema and a user's description, generate a SQL query that answers their request.

Database Schema:
${schemaDescription}

User's request: ${description}

Please respond with ONLY the SQL query, no explanation or markdown. The query should be ready to execute.`

      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 1000,
          messages: [
            {
              role: 'user',
              content: prompt,
            },
          ],
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)
        throw new Error(errorData?.error?.message || `API request failed: ${response.status}`)
      }

      const data = await response.json()
      const generatedQuery = data.content[0].text.trim()

      onQueryGenerated(generatedQuery)
      setDescription('')
    } catch (err) {
      console.error('Failed to generate query:', err)
      setError(err instanceof Error ? err.message : 'Failed to generate query')
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
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
          {!showApiKeyInput && (
            <button className="btn btn-sm btn-secondary" onClick={clearApiKey}>
              Clear API Key
            </button>
          )}
        </div>

        {showApiKeyInput ? (
          <div className="api-key-section">
            <label className="api-key-label">Anthropic API Key Required</label>
            <input
              type="password"
              className="api-key-input"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-api03-..."
              onKeyDown={(e) => e.key === 'Enter' && saveApiKey()}
            />
            <div className="api-key-buttons">
              <button className="btn btn-sm btn-primary" onClick={saveApiKey} disabled={!apiKey.trim()}>
                Save Key
              </button>
              <button className="btn btn-sm btn-secondary" onClick={() => setShowApiKeyInput(false)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <input
              type="text"
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
              {schemasLoading && <span className="ai-status">Loading schemas...</span>}
            </div>
          </>
        )}

        {error && <div className="error-message">{error}</div>}
      </div>
    </>
  )
}
