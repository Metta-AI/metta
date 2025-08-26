import linksConfig from '../links.yaml'
import './App.css'

interface Link {
  name: string
  url: string
  short_url?: string
}

function App() {
  const links = linksConfig.links

  return (
    <div className="container">
      <h1>Softmax Research</h1>
      <p className="lead">Quick links to our main resources.</p>
      <ul className="links">
        {links.map((link: Link, index: number) => (
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
    </div>
  )
}

export default App
