import { useState, useEffect } from 'react'
import { Repo, TableInfo, TableSchema, SQLQueryResponse } from './repo'

const SQL_QUERY_CSS = `
  .sql-query-container {
    display: flex;
    gap: 20px;
    height: calc(100vh - 80px);
    padding: 20px;
  }

  .tables-sidebar {
    width: 250px;
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    overflow-y: auto;
  }

  .tables-sidebar h3 {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 18px;
    color: #333;
  }

  .table-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .table-item {
    padding: 8px 12px;
    margin-bottom: 5px;
    background-color: white;
    border-radius: 4px;
    cursor: pointer;
    border: 1px solid #e0e0e0;
    transition: all 0.2s ease;
  }

  .table-item:hover {
    background-color: #e8f4f8;
    border-color: #2196F3;
  }

  .table-item.selected {
    background-color: #2196F3;
    color: white;
    border-color: #1976D2;
  }

  .table-name {
    font-weight: 500;
    margin-bottom: 2px;
  }

  .table-info {
    font-size: 12px;
    color: #666;
  }

  .table-item.selected .table-info {
    color: #e3f2fd;
  }

  .query-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .query-input-section {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .query-input-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }

  .query-input-header h3 {
    margin: 0;
    font-size: 18px;
    color: #333;
  }

  .query-textarea {
    width: 100%;
    min-height: 150px;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 14px;
    resize: vertical;
  }

  .query-textarea:focus {
    outline: none;
    border-color: #2196F3;
    box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
  }

  .results-section {
    flex: 1;
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }

  .results-header h3 {
    margin: 0;
    font-size: 18px;
    color: #333;
  }

  .results-info {
    font-size: 14px;
    color: #666;
  }

  .results-table-container {
    flex: 1;
    overflow: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
  }

  .results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }

  .results-table th {
    background-color: #f5f5f5;
    padding: 10px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #ddd;
    position: sticky;
    top: 0;
    z-index: 1;
  }

  .results-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #eee;
  }

  .results-table tr:hover {
    background-color: #f8f9fa;
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
    background-color: #f0f7ff;
    border-radius: 4px;
    padding: 15px;
    margin-bottom: 20px;
    border: 1px solid #b3d9ff;
  }

  .schema-info h4 {
    margin: 0 0 10px 0;
    color: #0066cc;
  }

  .schema-columns {
    font-size: 13px;
    line-height: 1.6;
  }

  .schema-column {
    margin-bottom: 4px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  .column-type {
    color: #666;
    font-size: 12px;
  }

  .btn {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .btn-primary {
    background-color: #2196F3;
    color: white;
  }

  .btn-primary:hover {
    background-color: #1976D2;
  }

  .btn-primary:disabled {
    background-color: #ccc;
    cursor: not-allowed;
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
  const [tables, setTables] = useState<TableInfo[]>([])
  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [tableSchema, setTableSchema] = useState<TableSchema | null>(null)
  const [query, setQuery] = useState('')
  const [queryState, setQueryState] = useState<QueryState>({ type: 'idle' })
  const [tablesLoading, setTablesLoading] = useState(true)
  const [schemaLoading, setSchemaLoading] = useState(false)

  useEffect(() => {
    loadTables()
  }, [repo])

  useEffect(() => {
    if (selectedTable) {
      loadTableSchema(selectedTable)
    } else {
      setTableSchema(null)
    }
  }, [selectedTable, repo])

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

  async function executeQuery() {
    if (!query.trim()) return

    try {
      setQueryState({ type: 'loading' })
      const result = await repo.executeQuery({ query })
      setQueryState({ type: 'success', data: result })
    } catch (error) {
      setQueryState({ 
        type: 'error', 
        error: error instanceof Error ? error.message : 'Query execution failed' 
      })
    }
  }

  function handleTableClick(tableName: string) {
    setSelectedTable(tableName)
    // Pre-fill query with a simple SELECT statement
    setQuery(`SELECT * FROM ${tableName} LIMIT 100`)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      executeQuery()
    }
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
              {tables.map(table => (
                <li
                  key={table.table_name}
                  className={`table-item ${selectedTable === table.table_name ? 'selected' : ''}`}
                  onClick={() => handleTableClick(table.table_name)}
                >
                  <div className="table-name">{table.table_name}</div>
                  <div className="table-info">
                    {table.column_count} columns • {table.row_count.toLocaleString()} rows
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="query-area">
          <div className="query-input-section">
            <div className="query-input-header">
              <h3>SQL Query</h3>
              <button 
                className="btn btn-primary" 
                onClick={executeQuery}
                disabled={!query.trim() || queryState.type === 'loading'}
              >
                {queryState.type === 'loading' ? 'Executing...' : 'Execute Query'}
              </button>
            </div>

            {tableSchema && !schemaLoading && (
              <div className="schema-info">
                <h4>Schema for {tableSchema.table_name}</h4>
                <div className="schema-columns">
                  {tableSchema.columns.map(col => (
                    <div key={col.name} className="schema-column">
                      <strong>{col.name}</strong>
                      <span className="column-type"> ({col.type}{col.nullable ? ', nullable' : ''})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <textarea
              className="query-textarea"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your SQL query here... (Cmd/Ctrl + Enter to execute)"
              spellCheck={false}
            />
          </div>

          <div className="results-section">
            <div className="results-header">
              <h3>Results</h3>
              {queryState.type === 'success' && (
                <div className="results-info">
                  {queryState.data.row_count} rows returned
                </div>
              )}
            </div>

            {queryState.type === 'error' && (
              <div className="error-message">
                <strong>Error:</strong> {queryState.error}
              </div>
            )}

            {queryState.type === 'loading' && (
              <div className="loading-message">Executing query...</div>
            )}

            {queryState.type === 'success' && queryState.data.row_count === 0 && (
              <div className="empty-results">No results returned</div>
            )}

            {queryState.type === 'success' && queryState.data.row_count > 0 && (
              <div className="results-table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      {queryState.data.columns.map(col => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {queryState.data.rows.map((row, idx) => (
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
              <div className="empty-results">
                Select a table or enter a query to see results
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}