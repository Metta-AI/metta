import { useState } from 'react'
import styles from './TrainingRuns.module.css'

interface TagEditorProps {
  tags: string[]
  canEdit: boolean
  onTagsChange: (newTags: string[]) => void
  onError: (error: string) => void
  disabled?: boolean
  compact?: boolean // For table rows vs detail pages
}

export function TagEditor({ tags, canEdit, onTagsChange, onError, disabled = false, compact = false }: TagEditorProps) {
  const [addingTag, setAddingTag] = useState(false)
  const [newTag, setNewTag] = useState('')
  const [confirmDeleteTag, setConfirmDeleteTag] = useState<string | null>(null)

  const handleStartAddingTag = () => {
    setAddingTag(true)
    setNewTag('')
  }

  const handleAddTag = async () => {
    const trimmedTag = newTag.trim()
    if (!trimmedTag) return

    if (tags.includes(trimmedTag)) {
      onError('Tag already exists')
      return
    }

    try {
      const updatedTags = [...tags, trimmedTag]
      await onTagsChange(updatedTags)
      setAddingTag(false)
      setNewTag('')
    } catch (err: any) {
      onError(`Failed to add tag: ${err.message}`)
    }
  }

  const handleCancelAddTag = () => {
    setAddingTag(false)
    setNewTag('')
  }

  const handleRemoveTagClick = (tag: string) => {
    setConfirmDeleteTag(tag)
  }

  const handleConfirmRemoveTag = async () => {
    if (!confirmDeleteTag) return

    try {
      const updatedTags = tags.filter((t) => t !== confirmDeleteTag)
      await onTagsChange(updatedTags)
      setConfirmDeleteTag(null)
    } catch (err: any) {
      onError(`Failed to remove tag: ${err.message}`)
    }
  }

  const handleCancelRemoveTag = () => {
    setConfirmDeleteTag(null)
  }

  const tagClassName = canEdit ? styles.tagRemovable : styles.tag
  const containerClassName = compact ? styles.tagsDisplay : styles.tagsDisplay

  return (
    <div>
      {/* Tags Display */}
      <div className={containerClassName}>
        {tags.length > 0 ? (
          tags.map((tag, index) => (
            <span key={index} className={tagClassName}>
              {tag}
              {canEdit && (
                <button
                  onClick={() => handleRemoveTagClick(tag)}
                  className={styles.removeTagBtn}
                  disabled={disabled}
                  title="Remove tag"
                >
                  ×
                </button>
              )}
            </span>
          ))
        ) : (
          <span style={{ color: '#999', fontStyle: 'italic' }}>No tags</span>
        )}
      </div>

      {/* Add Tag Section */}
      {canEdit && (
        <div>
          {addingTag ? (
            <div className={styles.addTagForm}>
              <div className={styles.tagInputContainer}>
                <input
                  type="text"
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  className={styles.tagInput}
                  placeholder="Enter tag name..."
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleAddTag()
                    } else if (e.key === 'Escape') {
                      handleCancelAddTag()
                    }
                  }}
                  autoFocus
                  disabled={disabled}
                />
                <div className={styles.tagInputActions}>
                  <button onClick={handleAddTag} disabled={!newTag.trim() || disabled} className={styles.saveBtn}>
                    Save
                  </button>
                  <button onClick={handleCancelAddTag} disabled={disabled} className={styles.cancelBtn}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <button onClick={handleStartAddingTag} className={styles.addTagBtnMain} disabled={disabled}>
              Add Tag
            </button>
          )}
        </div>
      )}

      {/* Confirmation Dialog */}
      {confirmDeleteTag && (
        <div className={styles.confirmDeleteContainer}>
          <div className={styles.confirmDeleteText}>Remove tag "{confirmDeleteTag}"?</div>
          <div className={styles.confirmDeleteActions}>
            <button onClick={handleConfirmRemoveTag} disabled={disabled} className={styles.confirmDeleteBtn}>
              Remove
            </button>
            <button onClick={handleCancelRemoveTag} disabled={disabled} className={styles.confirmCancelBtn}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
