import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Repo, SavedDashboard } from './repo'

// CSS for saved dashboards
const SAVED_DASHBOARDS_CSS = `
.dashboard-list {
  max-width: 800px;
  margin: 0 auto;
}

.dashboard-item {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
  transition: all 0.2s ease;
}

.dashboard-item:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,.15);
  transform: translateY(-2px);
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.dashboard-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0;
}

.dashboard-actions {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 6px 12px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn-primary {
  background: #007bff;
  color: #fff;
}

.btn-primary:hover {
  background: #0056b3;
}

.btn-danger {
  background: #dc3545;
  color: #fff;
}

.btn-danger:hover {
  background: #c82333;
}

.btn-secondary {
  background: #6c757d;
  color: #fff;
}

.btn-secondary:hover {
  background: #5a6268;
}

.dashboard-description {
  color: #666;
  margin-bottom: 12px;
  font-size: 14px;
}

.dashboard-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.meta-item {
  display: flex;
  flex-direction: column;
}

.meta-label {
  font-size: 12px;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.meta-value {
  font-size: 14px;
  color: #333;
  font-weight: 500;
}

.dashboard-dates {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #999;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #666;
}

.empty-state h3 {
  margin-bottom: 12px;
  color: #333;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error {
  text-align: center;
  padding: 40px;
  color: #dc3545;
}

/* Toast notification styles */
.toast-container {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  pointer-events: none;
}

.toast {
  background: #28a745;
  color: white;
  padding: 12px 20px;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 500;
  opacity: 0;
  transform: translateX(100%);
  transition: all 0.3s ease;
  pointer-events: auto;
  max-width: 300px;
  word-wrap: break-word;
}

.toast.show {
  opacity: 1;
  transform: translateX(0);
}

.toast.hide {
  opacity: 0;
  transform: translateX(100%);
}
`

interface SavedDashboardsProps {
  repo: Repo
  currentUser: string
}

export function SavedDashboards({ repo, currentUser }: SavedDashboardsProps) {
  const [dashboards, setDashboards] = useState<SavedDashboard[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<{
    message: string
    visible: boolean
  } | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    loadDashboards()
  }, [])

  const showToast = (message: string) => {
    setToast({ message, visible: true })
    setTimeout(() => {
      setToast((prev) => (prev ? { ...prev, visible: false } : null))
    }, 3000)
  }

  const loadDashboards = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await repo.listSavedDashboards()
      setDashboards(response.dashboards)
    } catch (err: any) {
      setError(err.message || 'Failed to load saved dashboards')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (dashboardId: string) => {
    if (!confirm('Are you sure you want to delete this dashboard?')) {
      return
    }

    try {
      await repo.deleteSavedDashboard(dashboardId)
      setDashboards(dashboards.filter((d) => d.id !== dashboardId))
    } catch (err: any) {
      alert(`Failed to delete dashboard: ${err.message}`)
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString()
  }

  const getShareableUrl = (dashboardId: string) => {
    const currentUrl = new URL(window.location.href)
    currentUrl.pathname = '/dashboard'
    currentUrl.searchParams.set('saved_id', dashboardId)
    return currentUrl.toString()
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      showToast('URL copied to clipboard!')
    } catch (err) {
      console.error('Failed to copy to clipboard:', err)
    }
  }

  const handleSelectDashboard = (dashboard: SavedDashboard) => {
    const url = new URL(window.location.href)
    url.searchParams.set('saved_id', dashboard.id)
    url.pathname = '/dashboard'
    navigate(url.pathname + url.search)
  }

  if (loading) {
    return (
      <div style={{ padding: '20px' }}>
        <style>{SAVED_DASHBOARDS_CSS}</style>
        <div className="loading">
          <h3>Loading saved dashboards...</h3>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '20px' }}>
        <style>{SAVED_DASHBOARDS_CSS}</style>
        <div className="error">
          <h3>Error loading dashboards</h3>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={loadDashboards}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      style={{
        padding: '20px',
        background: '#f8f9fa',
        minHeight: 'calc(100vh - 60px)',
      }}
    >
      <style>{SAVED_DASHBOARDS_CSS}</style>

      {/* Toast notification */}
      <div className="toast-container">
        {toast && <div className={`toast ${toast.visible ? 'show' : 'hide'}`}>{toast.message}</div>}
      </div>

      <div className="dashboard-list">
        <h1 style={{ marginBottom: '30px', color: '#333', textAlign: 'center' }}>Saved Dashboards</h1>

        {dashboards.length === 0 ? (
          <div className="empty-state">
            <h3>No saved dashboards</h3>
            <p>You haven't saved any dashboards yet. Create one from the main dashboard view.</p>
          </div>
        ) : (
          dashboards.map((dashboard) => (
            <div key={dashboard.id} className="dashboard-item">
              <div className="dashboard-header">
                <h3 className="dashboard-title">{dashboard.name}</h3>
                <div className="dashboard-actions">
                  <button className="btn btn-primary" onClick={() => handleSelectDashboard(dashboard)}>
                    Open
                  </button>
                  <button className="btn btn-secondary" onClick={() => copyToClipboard(getShareableUrl(dashboard.id))}>
                    Share
                  </button>
                  {dashboard.user_id === currentUser && (
                    <button className="btn btn-danger" onClick={() => handleDelete(dashboard.id)}>
                      Delete
                    </button>
                  )}
                </div>
              </div>

              {dashboard.description && <div className="dashboard-description">{dashboard.description}</div>}

              <div className="dashboard-meta">
                <div className="meta-item">
                  <div className="meta-label">Created At</div>
                  <div className="meta-value">{formatDate(dashboard.created_at)}</div>
                </div>
                <div className="meta-item">
                  <div className="meta-label">Created By</div>
                  <div className="meta-value">{dashboard.user_id}</div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
