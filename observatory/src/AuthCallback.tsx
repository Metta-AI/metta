import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import { initiateLogin, setToken } from './auth'

export function AuthCallback() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [status, setStatus] = useState<'processing' | 'success' | 'error'>('processing')
  const [errorMessage, setErrorMessage] = useState<string>('')

  useEffect(() => {
    const token = searchParams.get('token')

    if (!token) {
      setStatus('error')
      setErrorMessage('No token received from authentication server')
      return
    }

    try {
      // Save the token
      setToken(token)
      setStatus('success')

      // Redirect to home page after a short delay
      const timeoutId = setTimeout(() => {
        navigate('/')
      }, 2000)

      return () => clearTimeout(timeoutId)
    } catch (error) {
      setStatus('error')
      setErrorMessage(error instanceof Error ? error.message : 'Failed to save authentication token')
    }
  }, [searchParams, navigate])

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#fffdf4',
        fontFamily: '"ABC Marfa Variable", "Roboto", -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      <div
        style={{
          width: 'min(560px, 100%)',
          background: 'rgba(255, 254, 248, 0.95)',
          borderRadius: '24px',
          border: '1px solid rgba(14, 39, 88, 0.12)',
          boxShadow: '0 32px 60px rgba(14, 39, 88, 0.12)',
          padding: 'clamp(2rem, 6vw, 3.25rem)',
          textAlign: 'center',
        }}
      >
        {status === 'processing' && (
          <>
            <div
              style={{
                height: '76px',
                width: '76px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '2.5rem',
                margin: '0 auto 20px',
                border: '1px solid rgba(14, 39, 88, 0.12)',
                background: 'rgba(14, 39, 88, 0.08)',
              }}
            >
              ⏳
            </div>
            <h1 style={{ fontSize: '2.35rem', fontWeight: 600, marginBottom: '12px' }}>Processing...</h1>
            <p style={{ color: 'rgba(14, 39, 88, 0.72)', marginBottom: '12px' }}>Saving your authentication token</p>
          </>
        )}

        {status === 'success' && (
          <>
            <div
              style={{
                height: '76px',
                width: '76px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '2.5rem',
                margin: '0 auto 20px',
                border: '1px solid rgba(14, 39, 88, 0.12)',
                background: 'rgba(26, 107, 63, 0.16)',
                color: '#195C38',
              }}
            >
              ✓
            </div>
            <h1 style={{ fontSize: '2.35rem', fontWeight: 600, marginBottom: '12px' }}>You're all set!</h1>
            <p style={{ color: 'rgba(14, 39, 88, 0.72)', marginBottom: '12px' }}>
              Authentication complete. Redirecting you now...
            </p>
          </>
        )}

        {status === 'error' && (
          <>
            <div
              style={{
                height: '76px',
                width: '76px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '2.5rem',
                margin: '0 auto 20px',
                border: '1px solid rgba(14, 39, 88, 0.12)',
                background: 'rgba(176, 46, 38, 0.16)',
                color: '#952F2B',
              }}
            >
              ⚠
            </div>
            <h1 style={{ fontSize: '2.35rem', fontWeight: 600, marginBottom: '12px' }}>Something went wrong</h1>
            <p style={{ color: 'rgba(14, 39, 88, 0.72)', marginBottom: '12px' }}>{errorMessage}</p>
            <p style={{ color: 'rgba(14, 39, 88, 0.72)', marginBottom: '12px' }}>
              Please retry the login process or contact support if the issue persists.
            </p>
            <button
              onClick={() => initiateLogin()}
              style={{
                appearance: 'none',
                borderRadius: '999px',
                border: '2px solid #0E2758',
                background: '#0E2758',
                color: '#fffdf4',
                cursor: 'pointer',
                padding: '0.9rem 1.8rem',
                fontSize: '0.95rem',
                fontFamily: '"Marfa Mono", "Courier New", monospace',
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginTop: '32px',
              }}
            >
              Retry Login
            </button>
          </>
        )}
      </div>
    </div>
  )
}
