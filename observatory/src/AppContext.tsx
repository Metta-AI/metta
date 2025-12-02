import { createContext, PropsWithChildren, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { config } from './config'
import { Repo } from './repo'

export const AppContext = createContext<{
  repo: Repo
  currentUser: string
}>({
  repo: new Repo(),
  currentUser: '',
})

export const AppProvider = ({ children }: PropsWithChildren) => {
  // Data loading state
  type DefaultState = {
    type: 'error'
    error: string | null
  }
  type LoadingState = {
    type: 'loading'
  }
  type RepoState = {
    type: 'repo'
    repo: Repo
    currentUser: string
  }
  type State = DefaultState | LoadingState | RepoState

  const [state, setState] = useState<State>({ type: 'loading' })
  const navigate = useNavigate()

  useEffect(() => {
    const initializeRepo = async () => {
      const serverUrl = config.apiBaseUrl
      try {
        const repo = new Repo(serverUrl)

        // Get current user
        const userInfo = await repo.whoami()
        const currentUser = userInfo.user_email

        setState({ type: 'repo', repo, currentUser })
      } catch (err: any) {
        setState({
          type: 'error',
          error: `Failed to connect to server: ${err.message}. Make sure the server is running at ${serverUrl}`,
        })
      }
    }

    initializeRepo()
  }, [navigate])

  switch (state.type) {
    case 'error':
      return (
        <div
          style={{
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
            <p style={{ marginBottom: '20px', color: '#666' }}>Unable to connect to the evaluation server.</p>
            {state.error && <div style={{ color: 'red', marginTop: '10px', marginBottom: '20px' }}>{state.error}</div>}
            <p style={{ color: '#666', fontSize: '14px' }}>Please ensure the server is running and accessible.</p>
          </div>
        </div>
      )

    case 'loading':
      return (
        <div
          style={{
            margin: 0,
            padding: '20px',
            background: '#f8f9fa',
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div className="text-center text-gray-600">
            <h2>Connecting to server...</h2>
            <p>Loading evaluation data from the server.</p>
          </div>
        </div>
      )

    case 'repo':
      return (
        <AppContext.Provider value={{ repo: state.repo, currentUser: state.currentUser }}>
          {children}
        </AppContext.Provider>
      )
    default:
      throw new Error(`Unknown state type: ${state satisfies never}`)
  }
}
