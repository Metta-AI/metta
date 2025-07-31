import { useState, useEffect } from 'react'
import { SavedDashboardCreate } from './repo'

// CSS for modal
const MODAL_CSS = `
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  max-width: 500px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #eee;
}

.modal-title {
  font-size: 20px;
  font-weight: 600;
  color: #333;
  margin: 0;
}

.modal-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #999;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.modal-close:hover {
  background: #f5f5f5;
  color: #666;
}

.form-group {
  margin-bottom: 20px;
}

.form-label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #333;
  margin-bottom: 8px;
}

.form-input {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.2s ease;
  box-sizing: border-box;
}

.form-input:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.form-textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  min-height: 100px;
  resize: vertical;
  transition: border-color 0.2s ease;
  box-sizing: border-box;
  font-family: inherit;
}

.form-textarea:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.modal-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
  min-width: 80px;
}

.btn-primary {
  background: #007bff;
  color: #fff;
}

.btn-primary:hover {
  background: #0056b3;
}

.btn-primary:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.btn-secondary {
  background: #6c757d;
  color: #fff;
}

.btn-secondary:hover {
  background: #545b62;
}

.save-status {
  margin-top: 16px;
  padding: 12px;
  border-radius: 6px;
  font-size: 14px;
}

.save-status.success {
  background: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
}

.save-status.error {
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

.required {
  color: #dc3545;
}
`

interface SaveDashboardModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (dashboardData: SavedDashboardCreate) => Promise<void>
  initialName?: string
  initialDescription?: string
  isUpdate?: boolean
}

export function SaveDashboardModal({
  isOpen,
  onClose,
  onSave,
  initialName = '',
  initialDescription = '',
  isUpdate = false,
}: SaveDashboardModalProps) {
  const [name, setName] = useState(initialName)
  const [description, setDescription] = useState(initialDescription)
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<{
    type: 'success' | 'error'
    message: string
  } | null>(null)

  // Reset form when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setName(initialName)
      setDescription(initialDescription)
      setSaveStatus(null)
    }
  }, [isOpen, initialName, initialDescription])

  const handleSave = async () => {
    if (!name.trim()) {
      setSaveStatus({ type: 'error', message: 'Dashboard name is required' })
      return
    }

    setIsSaving(true)
    setSaveStatus(null)

    try {
      const dashboardData: SavedDashboardCreate = {
        name: name.trim(),
        description: description.trim() || undefined,
        type: 'scorecard',
        dashboard_state: {}, // This will be filled by the parent component
      }

      await onSave(dashboardData)
      setSaveStatus({
        type: 'success',
        message: `Dashboard ${isUpdate ? 'updated' : 'saved'} successfully!`,
      })

      // Close modal after a short delay
      setTimeout(() => {
        onClose()
        setSaveStatus(null)
      }, 1500)
    } catch (err: any) {
      setSaveStatus({
        type: 'error',
        message: err.message || `Failed to ${isUpdate ? 'update' : 'save'} dashboard`,
      })
    } finally {
      setIsSaving(false)
    }
  }

  const handleClose = () => {
    if (!isSaving) {
      onClose()
      setSaveStatus(null)
    }
  }

  if (!isOpen) {
    return null
  }

  return (
    <>
      <style>{MODAL_CSS}</style>
      <div className="modal-overlay" onClick={handleClose}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h2 className="modal-title">{isUpdate ? 'Update Dashboard' : 'Save Dashboard'}</h2>
            <button className="modal-close" onClick={handleClose} disabled={isSaving}>
              ×
            </button>
          </div>

          <div className="form-group">
            <label className="form-label">
              Dashboard Name <span className="required">*</span>
            </label>
            <input
              type="text"
              className="form-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter dashboard name"
              disabled={isSaving}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Description</label>
            <textarea
              className="form-textarea"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description"
              disabled={isSaving}
            />
          </div>

          {saveStatus && <div className={`save-status ${saveStatus.type}`}>{saveStatus.message}</div>}

          <div className="modal-actions">
            <button className="btn btn-secondary" onClick={handleClose} disabled={isSaving}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleSave} disabled={isSaving || !name.trim()}>
              {isSaving ? 'Saving...' : isUpdate ? 'Update' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
