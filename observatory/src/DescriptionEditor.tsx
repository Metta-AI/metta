import { useState } from 'react'
import styles from './TrainingRuns.module.css'

interface DescriptionEditorProps {
  description: string | null
  canEdit: boolean
  onDescriptionChange: (newDescription: string) => void
  onError: (error: string) => void
  disabled?: boolean
  compact?: boolean // For table rows vs detail pages
  placeholder?: string
}

export function DescriptionEditor({
  description,
  canEdit,
  onDescriptionChange,
  onError,
  disabled = false,
  compact = false,
  placeholder = 'Enter description...',
}: DescriptionEditorProps) {
  const [editing, setEditing] = useState(false)
  const [editValue, setEditValue] = useState('')

  const handleStartEditing = () => {
    setEditing(true)
    setEditValue(description || '')
  }

  const handleSave = async () => {
    try {
      await onDescriptionChange(editValue)
      setEditing(false)
      setEditValue('')
    } catch (err: any) {
      onError(`Failed to update description: ${err.message}`)
    }
  }

  const handleCancel = () => {
    setEditing(false)
    setEditValue('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSave()
    } else if (e.key === 'Escape') {
      handleCancel()
    }
  }

  if (editing) {
    return (
      <div className={styles.editDescriptionForm}>
        <textarea
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          className={styles.editDescriptionInput}
          placeholder={placeholder}
          rows={compact ? 3 : 4}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <div className={styles.editDescriptionActions}>
          <button onClick={handleSave} disabled={disabled} className={styles.saveBtn}>
            Save
          </button>
          <button onClick={handleCancel} disabled={disabled} className={styles.cancelBtn}>
            Cancel
          </button>
        </div>
      </div>
    )
  }

  // Display mode
  return (
    <div>
      <span className={compact ? styles.trainingRunDescription : undefined}>
        {description || (compact ? '—' : 'No description provided')}
      </span>
      {canEdit && (
        <button onClick={handleStartEditing} className={styles.editDescriptionBtn} disabled={disabled}>
          Edit
        </button>
      )}
    </div>
  )
}
