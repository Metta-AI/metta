import { useEffect, useState } from 'react'
import { loadDataFromUri, loadDataFromFile } from './data_loader'
import { DataRepo, Repo } from './repo'
import { Dashboard } from './Dashboard'

function App() {
  // Data loading state
  type DefaultState = {
    type: 'default'
    error: string | null
  }
  type LoadingState = {
    type: 'loading'
    dataUri: string
  }
  type RepoState = {
    type: 'repo'
    repo: Repo
  }
  type State = DefaultState | LoadingState | RepoState

  const [state, setState] = useState<State>({ type: 'default', error: null })

  const getDataUri: () => string | null = () => {
    const params = new URLSearchParams(window.location.search)
    return params.get('data')
  }

  useEffect(() => {
    const loadData = async () => {
      const dataUri = getDataUri()
      if (!dataUri) {
        return
      }

      setState({ type: 'loading', dataUri })

      try {
        const data = await loadDataFromUri(dataUri)
        setState({ type: 'repo', repo: new DataRepo(data) })
      } catch (err: any) {
        setState({ type: 'default', error: err.message })
      }
    }

    loadData()
  }, [])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const data = await loadDataFromFile(file)
      setState({ type: 'repo', repo: new DataRepo(data) })
    } catch (err: any) {
      setState({
        type: 'default',
        error: 'Failed to load data from file: ' + err.message,
      })
    }
  }

  if (state.type === 'default') {
    return (
      <div
        style={{
          fontFamily: 'Arial, sans-serif',
          margin: 0,
          padding: '20px',
          background: '#f8f9fa',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            maxWidth: '600px',
            margin: '0 auto',
            background: '#fff',
            padding: '40px',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,.1)',
            textAlign: 'center',
          }}
        >
          <h1
            style={{
              color: '#333',
              marginBottom: '20px',
            }}
          >
            Policy Evaluation Dashboard
          </h1>
          <p style={{ marginBottom: '20px', color: '#666' }}>
            Upload your evaluation data or provide a data URI as a query parameter.
          </p>
          <div style={{ marginBottom: '20px' }}>
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              style={{
                padding: '10px',
                border: '2px dashed #ddd',
                borderRadius: '4px',
                width: '100%',
                cursor: 'pointer',
              }}
            />
          </div>
          {state.error && <div style={{ color: 'red', marginTop: '10px' }}>{state.error}</div>}
          <p style={{ color: '#666', fontSize: '14px' }}>
            Or add <code>?data=YOUR_DATA_URI</code> to the URL
          </p>
        </div>
      </div>
    )
  }

  if (state.type === 'loading') {
    return <div>Loading data...</div>
  }

  if (state.type === 'repo') {
    return <Dashboard repo={state.repo} />
  }

  return null
}

export default App
