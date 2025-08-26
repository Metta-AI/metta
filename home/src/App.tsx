import { useState, useEffect } from 'react'
import linksConfig from '../links.yaml'
import './App.css'

interface Link {
  name: string
  url: string
  short_url?: string
}

interface Config {
  links: Link[]
  host: string
}

function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const isDev = import.meta.env.DEV

  useEffect(() => {
    if (isDev) {
      // Development: use imported YAML directly
      setConfig({
        links: linksConfig.links,
        host: 'localhost',
      })
      setLoading(false)
    } else {
      // Production: fetch runtime config from Helm
      fetch('/config.json')
        .then((res) => {
          if (!res.ok) throw new Error('Failed to load configuration')
          return res.json()
        })
        .then((data) => {
          setConfig(data)
          setLoading(false)
        })
        .catch((err) => {
          setError(err.message)
          setLoading(false)
        })
    }
  }, [isDev])

  if (loading) {
    return (
      <div className="container">
        <div className="loading">Loading...</div>
      </div>
    )
  }

  if (error || !config) {
    return (
      <div className="container">
        <div className="error">Error loading configuration: {error}</div>
      </div>
    )
  }

  return (
    <div className="container">
      <h1>Softmax Research</h1>
      <p className="lead">Quick links to our main resources.</p>
      <ul className="links">
        {config.links.map((link, index) => (
          <li key={index}>
            <a className="card" href={link.url} target="_blank" rel="noopener noreferrer">
              <div className="title-row">
                <div>{link.name}</div>
                {link.short_url && <span className="short-url">/{link.short_url}</span>}
              </div>
              <small>{link.url}</small>
            </a>
          </li>
        ))}
      </ul>
      <footer>home: {config.host}</footer>
    </div>
  )
}

export default App
