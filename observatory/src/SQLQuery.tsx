import { useEffect, useState } from 'react'
import { AIQueryBuilder } from './AIQueryBuilder'
import type { Repo, SQLQueryResponse, TableInfo, TableSchema } from './repo'

interface QueryHistoryItem {
  query: string
  timestamp: number
  executionTime?: number
  rowCount?: number
  error?: boolean
}

const SQL_QUERY_CSS = `
  .sql-query-container {
    display: flex;
    gap: 16px;
    height: calc(100vh - 60px);
    padding: 16px;
    background-color: #fafafa;
  }

  .tables-sidebar {
    width: 240px;
    background-color: white;
    border-radius: 6px;
    padding: 16px;
    overflow-y: auto;
    border: 1px solid #e5e7eb;
  }

  .tables-sidebar h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .table-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .table-item {
    padding: 8px 10px;
    margin-bottom: 2px;
    background-color: transparent;
    border-radius: 4px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s ease;
  }

  .table-item:hover {
    background-color: #f3f4f6;
  }

  .table-item.selected {
    background-color: #3b82f6;
    color: white;
  }

  .table-name {
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 2px;
  }

  .table-info {
    font-size: 11px;
    color: #6b7280;
  }

  .table-item.selected .table-info {
    color: #dbeafe;
  }

  .query-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .query-input-section {
    background-color: white;
    border-radius: 6px;
    padding: 16px;
    border: 1px solid #e5e7eb;
    position: relative;
  }

  .query-input-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .query-input-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .query-input-wrapper {
    position: relative;
  }

  .query-textarea {
    width: 100%;
    min-height: 120px;
    padding: 10px;
    padding-bottom: 45px;
    padding-right: 140px;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 13px;
    resize: vertical;
    background-color: #f9fafb;
    box-sizing: border-box;
  }

  .query-textarea:focus {
    outline: none;
    border-color: #3b82f6;
    background-color: white;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .results-section {
    flex: 1;
    background-color: white;
    border-radius: 6px;
    padding: 16px;
    border: 1px solid #e5e7eb;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .results-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .results-info {
    font-size: 12px;
    color: #6b7280;
  }

  .results-table-container {
    flex: 1;
    overflow: auto;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
  }

  .results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  .results-table th {
    background-color: #f9fafb;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    color: #374151;
    border-bottom: 1px solid #e5e7eb;
    position: sticky;
    top: 0;
    z-index: 1;
    cursor: pointer;
    user-select: none;
    transition: background-color 0.15s;
  }

  .results-table th:hover {
    background-color: #f3f4f6;
  }

  .results-table th.sorted-asc::after {
    content: ' ↑';
    font-size: 11px;
    color: #3b82f6;
  }

  .results-table th.sorted-desc::after {
    content: ' ↓';
    font-size: 11px;
    color: #3b82f6;
  }

  .results-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #f3f4f6;
  }

  .results-table tr:hover {
    background-color: #f9fafb;
  }

  .error-message {
    background-color: #fee;
    color: #c00;
    padding: 15px;
    border-radius: 4px;
    border: 1px solid #fcc;
    margin-bottom: 15px;
  }

  .loading-message {
    text-align: center;
    padding: 40px;
    color: #666;
    font-size: 16px;
  }

  .empty-results {
    text-align: center;
    padding: 40px;
    color: #999;
  }

  .schema-info {
    background-color: #eff6ff;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 12px;
    border: 1px solid #dbeafe;
  }

  .schema-info h4 {
    margin: 0 0 8px 0;
    color: #1e40af;
    font-size: 12px;
    font-weight: 600;
  }

  .schema-columns {
    font-size: 12px;
    line-height: 1.5;
  }

  .schema-column {
    margin-bottom: 2px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  .column-type {
    color: #6b7280;
    font-size: 11px;
  }

  .btn {
    padding: 6px 14px;
    border: none;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .btn-primary {
    background-color: #3b82f6;
    color: white;
  }

  .btn-primary:hover {
    background-color: #2563eb;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  }

  .btn-primary:disabled {
    background-color: #e5e7eb;
    color: #9ca3af;
    cursor: not-allowed;
  }

  .execute-button {
    position: absolute;
    bottom: 10px;
    right: 10px;
    z-index: 2;
  }

  .query-history-section {
    margin-top: 24px;
    border-top: 1px solid #e5e7eb;
    padding-top: 16px;
  }

  .query-history-section h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .clear-history-btn {
    font-size: 11px;
    padding: 2px 6px;
    background-color: transparent;
    color: #6b7280;
    border: 1px solid #e5e7eb;
    text-transform: none;
    letter-spacing: normal;
    font-weight: 400;
  }

  .clear-history-btn:hover {
    background-color: #f3f4f6;
    color: #374151;
  }

  .history-list {
    list-style: none;
    padding: 0;
    margin: 0;
    max-height: 300px;
    overflow-y: auto;
  }

  .history-item {
    padding: 8px 10px;
    margin-bottom: 2px;
    background-color: transparent;
    border-radius: 4px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s ease;
    font-size: 12px;
  }

  .history-item:hover {
    background-color: #f3f4f6;
  }

  .history-query {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 11px;
    color: #374151;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 2px;
  }

  .history-meta {
    font-size: 10px;
    color: #6b7280;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .history-status {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .history-status.success {
    color: #10b981;
  }

  .history-status.error {
    color: #ef4444;
  }
`

interface Props {
  repo: Repo
}

type QueryState =
  | { type: 'idle' }
  | { type: 'loading' }
  | { type: 'success'; data: SQLQueryResponse }
  | { type: 'error'; error: string }

export function SQLQuery({ repo }: Props) {
  const [tables, setTables] = useState<Array<TableInfo>>([])
  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [tableSchema, setTableSchema] = useState<TableSchema | null>(null)
  const [query, setQuery] = useState('')
  const [queryState, setQueryState] = useState<QueryState>({ type: 'idle' })
  const [tablesLoading, setTablesLoading] = useState(true)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [sortConfig, setSortConfig] = useState<{ column: string; direction: 'asc' | 'desc' } | null>(null)
  const [queryHistory, setQueryHistory] = useState<Array<QueryHistoryItem>>([])

  const HISTORY_KEY = 'sql_query_history'
  const MAX_HISTORY_ITEMS = 50

  useEffect(() => {
    loadTables()
    loadQueryHistory()
  }, [])

  useEffect(() => {
    if (selectedTable) {
      loadTableSchema(selectedTable)
    } else {
      setTableSchema(null)
    }
  }, [selectedTable])

  async function loadTables() {
    try {
      setTablesLoading(true)
      const tableList = await repo.listTables()
      setTables(tableList)
    } catch (error) {
      console.error('Failed to load tables:', error)
    } finally {
      setTablesLoading(false)
    }
  }

  async function loadTableSchema(tableName: string) {
    try {
      setSchemaLoading(true)
      const schema = await repo.getTableSchema(tableName)
      setTableSchema(schema)
    } catch (error) {
      console.error('Failed to load table schema:', error)
    } finally {
      setSchemaLoading(false)
    }
  }

  function loadQueryHistory() {
    try {
      const stored = localStorage.getItem(HISTORY_KEY)
      if (stored) {
        const history = JSON.parse(stored) as Array<QueryHistoryItem>
        setQueryHistory(history)
      }
    } catch (error) {
      console.error('Failed to load query history:', error)
    }
  }

  function saveQueryToHistory(queryText: string, result: SQLQueryResponse | null, error = false) {
    const newItem: QueryHistoryItem = {
      query: queryText,
      timestamp: Date.now(),
      rowCount: result?.row_count,
      error,
    }

    const updatedHistory = [newItem, ...queryHistory.filter((item) => item.query !== queryText)].slice(
      0,
      MAX_HISTORY_ITEMS
    )

    setQueryHistory(updatedHistory)
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(updatedHistory))
    } catch (error) {
      console.error('Failed to save query history:', error)
    }
  }

  function clearHistory() {
    setQueryHistory([])
    try {
      localStorage.removeItem(HISTORY_KEY)
    } catch (error) {
      console.error('Failed to clear query history:', error)
    }
  }

  async function executeQuery() {
    if (!query.trim()) {
      return
    }

    try {
      setQueryState({ type: 'loading' })
      const result = await repo.executeQuery({ query })
      setQueryState({ type: 'success', data: result })
      saveQueryToHistory(query, result, false)
    } catch (error) {
      setQueryState({
        type: 'error',
        error: error instanceof Error ? error.message : 'Query execution failed',
      })
      saveQueryToHistory(query, null, true)
    }
  }

  function handleTableClick(tableName: string) {
    setSelectedTable(tableName)
    // Pre-fill query with a simple SELECT statement
    setQuery(`SELECT * FROM ${tableName} LIMIT 1000`)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      executeQuery()
    }
  }

  function handleSort(column: string) {
    if (queryState.type !== 'success') {
      return
    }

    let direction: 'asc' | 'desc' = 'asc'
    if (sortConfig && sortConfig.column === column && sortConfig.direction === 'asc') {
      direction = 'desc'
    }
    setSortConfig({ column, direction })
  }

  function getSortedRows() {
    if (queryState.type !== 'success' || !sortConfig) {
      return queryState.type === 'success' ? queryState.data.rows : []
    }

    const columnIndex = queryState.data.columns.indexOf(sortConfig.column)
    if (columnIndex === -1) {
      return queryState.data.rows
    }

    return [...queryState.data.rows].sort((a, b) => {
      const aVal = a[columnIndex]
      const bVal = b[columnIndex]

      // Handle null values
      if (aVal === null && bVal === null) {
        return 0
      }
      if (aVal === null) {
        return sortConfig.direction === 'asc' ? 1 : -1
      }
      if (bVal === null) {
        return sortConfig.direction === 'asc' ? -1 : 1
      }

      // Compare values
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal
      }

      // String comparison
      const aStr = String(aVal).toLowerCase()
      const bStr = String(bVal).toLowerCase()

      if (sortConfig.direction === 'asc') {
        return aStr < bStr ? -1 : aStr > bStr ? 1 : 0
      }
      return aStr > bStr ? -1 : aStr < bStr ? 1 : 0
    })
  }

  return (
    <>
      <style>{SQL_QUERY_CSS}</style>
      <div className="sql-query-container">
        <div className="tables-sidebar">
          <h3>Tables</h3>
          {tablesLoading ? (
            <div className="loading-message">Loading tables...</div>
          ) : (
            <ul className="table-list">
              {tables.map((table) => (
                <li
                  key={table.table_name}
                  className={`table-item ${selectedTable === table.table_name ? 'selected' : ''}`}
                  onClick={() => handleTableClick(table.table_name)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      handleTableClick(table.table_name)
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className="table-name">{table.table_name}</div>
                  <div className="table-info">
                    {table.column_count} columns • {table.row_count.toLocaleString()} rows
                  </div>
                </li>
              ))}
            </ul>
          )}

          {queryHistory.length > 0 && (
            <div className="query-history-section">
              <h3>
                Query History
                <button type="button" className="btn clear-history-btn" onClick={clearHistory}>
                  Clear
                </button>
              </h3>
              <ul className="history-list">
                {queryHistory.map((item, index) => {
                  const date = new Date(item.timestamp)
                  const timeStr = date.toLocaleTimeString()
                  const dateStr = date.toLocaleDateString()
                  const isToday = new Date().toDateString() === date.toDateString()

                  return (
                    <li 
                      key={index} 
                      className="history-item" 
                      onClick={() => setQuery(item.query)} 
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          setQuery(item.query)
                        }
                      }}
                      role="button"
                      tabIndex={0}
                      title={item.query}
                    >
                      <div className="history-query">{item.query}</div>
                      <div className="history-meta">
                        <span>{isToday ? timeStr : dateStr}</span>
                        <span className={`history-status ${item.error ? 'error' : 'success'}`}>
                          {item.error ? 'Error' : item.rowCount !== undefined ? `${item.rowCount} rows` : ''}
                        </span>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </div>

        <div className="query-area">
          <div className="query-input-section">
            <div className="query-input-header">
              <h3>SQL Query</h3>
            </div>

            {tableSchema && !schemaLoading && (
              <div className="schema-info">
                <h4>Schema for {tableSchema.table_name}</h4>
                <div className="schema-columns">
                  {tableSchema.columns.map((col) => (
                    <div key={col.name} className="schema-column">
                      <strong>{col.name}</strong>
                      <span className="column-type">
                        {' '}
                        ({col.type}
                        {col.nullable ? ', nullable' : ''})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <AIQueryBuilder repo={repo} onQueryGenerated={setQuery} />

            <div className="query-input-wrapper">
              <textarea
                className="query-textarea"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your SQL query here..."
                spellCheck={false}
              />
              <button
                className="btn btn-primary execute-button"
                onClick={executeQuery}
                disabled={!query.trim() || queryState.type === 'loading'}
              >
                {queryState.type === 'loading' ? (
                  'Executing...'
                ) : (
                  <>
                    Execute <span style={{ fontSize: '12px', opacity: 0.8 }}>⌘+Enter</span>
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="results-section">
            <div className="results-header">
              <h3>Results</h3>
              {queryState.type === 'success' && (
                <div className="results-info">{queryState.data.row_count} rows returned</div>
              )}
            </div>

            {queryState.type === 'error' && (
              <div className="error-message">
                <strong>Error:</strong> {queryState.error}
              </div>
            )}

            {queryState.type === 'loading' && <div className="loading-message">Executing query...</div>}

            {queryState.type === 'success' && queryState.data.row_count === 0 && (
              <div className="empty-results">No results returned</div>
            )}

            {queryState.type === 'success' && queryState.data.row_count > 0 && (
              <div className="results-table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      {queryState.data.columns.map((col) => (
                        <th
                          key={col}
                          onClick={() => handleSort(col)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              handleSort(col)
                            }
                          }}
                          role="button"
                          tabIndex={0}
                          className={
                            sortConfig?.column === col
                              ? sortConfig.direction === 'asc'
                                ? 'sorted-asc'
                                : 'sorted-desc'
                              : ''
                          }
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {getSortedRows().map((row, idx) => (
                      <tr key={idx}>
                        {row.map((cell, cellIdx) => (
                          <td key={cellIdx}>
                            {cell === null ? <em style={{ color: '#999' }}>NULL</em> : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {queryState.type === 'idle' && (
              <div className="empty-results">Select a table or enter a query to see results</div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
