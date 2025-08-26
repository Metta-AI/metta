import { useState, useEffect } from 'react'
import linksConfig from '../links.yaml'
import './App.css'

interface SubLink {
  name: string
  url: string
  short_urls: string[]
}

interface Link {
  name: string
  url: string
  icon?: string
  short_urls?: string[]
  sub_links?: SubLink[]
}

function getFaviconUrl(link: Link): string {
  // Use custom icon if provided
  if (link.icon) {
    return link.icon
  }

  // Otherwise try to load favicon from domain
  try {
    const urlObj = new URL(link.url)
    return `${urlObj.protocol}//${urlObj.hostname}/favicon.ico`
  } catch {
    return ''
  }
}

function cleanUrl(url: string): string {
  return url
    .replace(/^https?:\/\//, '') // Remove http:// or https://
    .replace(/\/$/, '') // Remove trailing slash
}

function App() {
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    const stored = localStorage.getItem('favorites')
    return stored ? new Set(JSON.parse(stored)) : new Set()
  })
  const [copiedText, setCopiedText] = useState<string | null>(null)
  const [isChrome, setIsChrome] = useState(false)

  useEffect(() => {
    localStorage.setItem('favorites', JSON.stringify(Array.from(favorites)))
  }, [favorites])

  useEffect(() => {
    // Check if browser is Chrome
    const userAgent = navigator.userAgent.toLowerCase()
    setIsChrome(userAgent.includes('chrome') && !userAgent.includes('edg'))
  }, [])

  const toggleFavorite = (linkName: string) => {
    setFavorites((prev) => {
      const newFavorites = new Set(prev)
      if (newFavorites.has(linkName)) {
        newFavorites.delete(linkName)
      } else {
        newFavorites.add(linkName)
      }
      return newFavorites
    })
  }

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedText(label)
      setTimeout(() => setCopiedText(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const copyableText = (text: string, label: string) => (
    <>
      <code className="copyable" onClick={() => copyToClipboard(text, label)} title="Click to copy">
        {text}
      </code>
      {copiedText === label && <span className="copied-indicator">Copied!</span>}
    </>
  )

  const sortedLinks = [...linksConfig.links].sort((a, b) => {
    const aFav = favorites.has(a.name)
    const bFav = favorites.has(b.name)
    if (aFav && !bFav) return -1
    if (!aFav && bFav) return 1
    return 0
  })

  return (
    <div className="container">
      <ul className="links">
        {sortedLinks.map((link: Link, index: number) => (
          <li key={index}>
            <div className="card-wrapper">
              <button
                className={`star-button ${favorites.has(link.name) ? 'starred' : ''}`}
                onClick={() => toggleFavorite(link.name)}
                aria-label={`${favorites.has(link.name) ? 'Unfavorite' : 'Favorite'} ${link.name}`}
              >
                {favorites.has(link.name) ? '★' : '☆'}
              </button>
              <a className="card" href={link.url} target="_blank" rel="noopener noreferrer">
                <div className="card-content">
                  <div className="favicon-wrapper">
                    {getFaviconUrl(link) ? (
                      <img
                        src={getFaviconUrl(link)}
                        alt=""
                        className="favicon"
                        onError={(e) => {
                          const wrapper = (e.target as HTMLImageElement).parentElement
                          if (wrapper) {
                            wrapper.classList.add('favicon-fallback')
                            wrapper.setAttribute('data-letter', link.name[0].toUpperCase())
                          }
                          ;(e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                    ) : (
                      <div className="favicon-fallback" data-letter={link.name[0].toUpperCase()}></div>
                    )}
                  </div>
                  <div className="card-text">
                    <div className="title-row">
                      <div>{link.name}</div>
                      {link.short_urls && link.short_urls.length > 0 && (
                        <span className="short-url">{link.short_urls.map((s) => `/${s}`).join(' ')}</span>
                      )}
                    </div>
                    <small>{cleanUrl(link.url)}</small>
                  </div>
                </div>
              </a>
            </div>
            {link.sub_links && link.sub_links.length > 0 && (
              <div className="sub-links">
                {link.sub_links.map((subLink: SubLink, subIndex: number) => (
                  <a key={subIndex} className="sub-link" href={subLink.url} target="_blank" rel="noopener noreferrer">
                    <span className="sub-link-name">{subLink.name}</span>
                    {subLink.short_urls && subLink.short_urls.length > 0 && (
                      <span className="sub-link-url">{subLink.short_urls.map((s) => `/${s}`).join(' ')}</span>
                    )}
                  </a>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>

      {isChrome && (
        <div className="setup-section">
          <div className="setup-card">
            <h2>Chrome Search</h2>

            <div className="setup-subsection">
              <ol>
                <li>Go to {copyableText('chrome://settings/searchEngines', 'settings')}</li>
                <li>Click "Add" under "Site search"</li>
                <li>
                  Enter:
                  <ul>
                    <li>Search engine: {copyableText('Softmax', 'name')}</li>
                    <li>Shortcut: {copyableText('s', 'shortcut')}</li>
                    <li>URL: {copyableText('https://home.softmax-research.net/%s', 'url')}</li>
                  </ul>
                </li>
              </ol>
            </div>

            <div className="setup-subsection">
              <p>
                Type <code>s</code> + [Space or Tab] in the address bar, then enter a shortcut like <code>g</code> for
                GitHub
              </p>
            </div>
          </div>
        </div>
      )}
      <div className="setup-section">
        <div className="setup-card">
          <h2>
            <code>metta go</code>
          </h2>

          <div className="setup-subsection">
            Run <code>metta go</code> from your command line to open a shortcut URL.
          </div>
        </div>
      </div>
      <div className="setup-section">
        <div className="setup-card">
          <h2>Adding links</h2>

          <div className="setup-subsection">
            Modify{' '}
            <a href="https://github.com/Metta-AI/metta/blob/main/home/links.yaml">
              <code>home/links.yaml</code>
            </a>
            . Updates should auto-deploy after merge.
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
