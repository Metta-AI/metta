import { useEffect, useState } from 'react'
import type { Repo, TokenCreate, TokenInfo } from './repo'

// CSS for token manager
const TOKEN_MANAGER_CSS = `
.token-manager {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 20px;
  background: #f8f9fa;
  min-height: 100vh;
}

.token-container {
  max-width: 800px;
  margin: 0 auto;
  background: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}

.token-header {
  color: #333;
  border-bottom: 1px solid #ddd;
  padding-bottom: 10px;
  margin-bottom: 20px;
}

.create-token-section {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 6px;
  margin-bottom: 30px;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: 500;
  color: #333;
}

.form-group input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  box-sizing: border-box;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
}

.btn-primary {
  background: #007bff;
  color: white;
}

.btn-primary:hover {
  background: #0056b3;
}

.btn-danger {
  background: #dc3545;
  color: white;
}

.btn-danger:hover {
  background: #c82333;
}

.btn:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.tokens-list {
  margin-top: 20px;
}

.token-item {
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 15px;
  margin-bottom: 10px;
  background: #fff;
}

.token-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.token-name {
  font-weight: 600;
  color: #333;
  font-size: 16px;
}

.token-id {
  font-family: monospace;
  font-size: 12px;
  color: #666;
  background: #f8f9fa;
  padding: 2px 6px;
  border-radius: 3px;
}

.token-details {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  font-size: 14px;
  color: #666;
}

.token-detail {
  display: flex;
  flex-direction: column;
}

.token-detail-label {
  font-weight: 500;
  color: #333;
  margin-bottom: 2px;
}

.token-detail-value {
  color: #666;
}

.error-message {
  color: #dc3545;
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 15px;
}

.success-message {
  color: #155724;
  background: #d4edda;
  border: 1px solid #c3e6cb;
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 15px;
}

.loading {
  text-align: center;
  color: #666;
  padding: 20px;
}

.empty-state {
  text-align: center;
  color: #666;
  padding: 40px 20px;
}

.empty-state h3 {
  margin-bottom: 10px;
  color: #333;
}
`

interface TokenManagerProps {
  repo: Repo
}

export function TokenManager({ repo }: TokenManagerProps) {
  const [tokens, setTokens] = useState<Array<TokenInfo>>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [newTokenName, setNewTokenName] = useState('')
  const [creatingToken, setCreatingToken] = useState(false)
  const [newlyCreatedToken, setNewlyCreatedToken] = useState<string | null>(null)

  useEffect(() => {
    loadTokens()
  }, [])

  const loadTokens = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await repo.listTokens()
      setTokens(response.tokens)
    } catch (err: any) {
      setError(`Failed to load tokens: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const createToken = async () => {
    if (!newTokenName.trim()) {
      setError('Token name is required')
      return
    }

    try {
      setCreatingToken(true)
      setError(null)
      setSuccessMessage(null)

      const tokenData: TokenCreate = { name: newTokenName.trim() }
      const response = await repo.createToken(tokenData)

      setNewlyCreatedToken(response.token)
      setSuccessMessage('Token created successfully! Copy the token below before closing this page.')
      setNewTokenName('')

      // Reload the tokens list
      await loadTokens()
    } catch (err: any) {
      setError(`Failed to create token: ${err.message}`)
    } finally {
      setCreatingToken(false)
    }
  }

  const deleteToken = async (tokenId: string) => {
    if (!confirm('Are you sure you want to delete this token? This action cannot be undone.')) {
      return
    }

    try {
      setError(null)
      await repo.deleteToken(tokenId)
      setSuccessMessage('Token deleted successfully')
      await loadTokens()
    } catch (err: any) {
      setError(`Failed to delete token: ${err.message}`)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setSuccessMessage('Token copied to clipboard!')
        setTimeout(() => setSuccessMessage(null), 2000)
      })
      .catch(() => {
        setError('Failed to copy token to clipboard')
      })
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  if (loading) {
    return (
      <div className="token-manager">
        <style>{TOKEN_MANAGER_CSS}</style>
        <div className="loading">Loading tokens...</div>
      </div>
    )
  }

  return (
    <div className="token-manager">
      <style>{TOKEN_MANAGER_CSS}</style>
      <div className="token-container">
        <h1 className="token-header">Machine Token Management</h1>

        {error && <div className="error-message">{error}</div>}
        {successMessage && <div className="success-message">{successMessage}</div>}

        <div className="create-token-section">
          <h3>Create New Token</h3>
          <div className="form-group">
            <label htmlFor="token-name">Token Name</label>
            <input
              id="token-name"
              type="text"
              value={newTokenName}
              onChange={(e) => setNewTokenName(e.target.value)}
              placeholder="Enter a descriptive name for this token"
              disabled={creatingToken}
            />
          </div>
          <button className="btn btn-primary" onClick={createToken} disabled={creatingToken || !newTokenName.trim()}>
            {creatingToken ? 'Creating...' : 'Create Token'}
          </button>
        </div>

        {newlyCreatedToken && (
          <div className="create-token-section">
            <h3>New Token Created</h3>
            <p style={{ marginBottom: '10px', color: '#666' }}>
              Copy this token and store it securely. You won't be able to see it again.
            </p>
            <div
              style={{
                background: '#f8f9fa',
                padding: '10px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontFamily: 'monospace',
                fontSize: '12px',
                wordBreak: 'break-all',
              }}
            >
              {newlyCreatedToken}
            </div>
            <button
              className="btn btn-primary"
              onClick={() => copyToClipboard(newlyCreatedToken)}
              style={{ marginTop: '10px' }}
            >
              Copy Token
            </button>
          </div>
        )}

        <div className="tokens-list">
          <h3>Your Tokens ({tokens.length})</h3>

          {tokens.length === 0 ? (
            <div className="empty-state">
              <h3>No tokens found</h3>
              <p>Create your first machine token above to get started.</p>
            </div>
          ) : (
            tokens.map((token) => (
              <div key={token.id} className="token-item">
                <div className="token-header-row">
                  <div>
                    <div className="token-name">{token.name}</div>
                    <div className="token-id">ID: {token.id}</div>
                  </div>
                  <button className="btn btn-danger" onClick={() => deleteToken(token.id)}>
                    Delete
                  </button>
                </div>
                <div className="token-details">
                  <div className="token-detail">
                    <span className="token-detail-label">Created</span>
                    <span className="token-detail-value">{formatDate(token.created_at)}</span>
                  </div>
                  <div className="token-detail">
                    <span className="token-detail-label">Expires</span>
                    <span className="token-detail-value">{formatDate(token.expiration_time)}</span>
                  </div>
                  <div className="token-detail">
                    <span className="token-detail-label">Last Used</span>
                    <span className="token-detail-value">
                      {token.last_used_at ? formatDate(token.last_used_at) : 'Never'}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
