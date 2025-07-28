import { useState } from 'react'
import { Link } from 'react-router-dom'
import { DescriptionEditor } from './DescriptionEditor'
import { TagEditor } from './TagEditor'
import styles from './TrainingRuns.module.css'
import type { Repo, TrainingRun } from './repo'

interface TrainingRunRowProps {
  run: TrainingRun
  canEdit: boolean
  repo: Repo
  onRunUpdate: (updatedRun: TrainingRun) => void
  onError: (error: string) => void
}

export function TrainingRunRow({ run, canEdit, repo, onRunUpdate, onError }: TrainingRunRowProps) {
  const [saving, setSaving] = useState(false)

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getStatusClass = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'running'
      case 'completed':
        return 'completed'
      case 'failed':
        return 'failed'
      default:
        return ''
    }
  }

  const handleDescriptionChange = async (newDescription: string) => {
    setSaving(true)
    try {
      const updatedRun = await repo.updateTrainingRunDescription(run.id, newDescription)
      onRunUpdate(updatedRun)
    } finally {
      setSaving(false)
    }
  }

  const handleTagsChange = async (newTags: Array<string>) => {
    setSaving(true)
    try {
      const updatedRun = await repo.updateTrainingRunTags(run.id, newTags)
      onRunUpdate(updatedRun)
    } finally {
      setSaving(false)
    }
  }

  return (
    <tr>
      <td>
        <Link to={`/training-run/${run.id}`} className={styles.trainingRunName}>
          {run.name}
        </Link>
      </td>

      <td className={styles.editDescriptionCell}>
        <DescriptionEditor
          description={run.description}
          canEdit={canEdit}
          onDescriptionChange={handleDescriptionChange}
          onError={onError}
          disabled={saving}
          compact={true}
        />
      </td>

      <td className={styles.tagsCell}>
        <TagEditor
          tags={run.tags}
          canEdit={canEdit}
          onTagsChange={handleTagsChange}
          onError={onError}
          disabled={saving}
          compact={true}
        />
      </td>

      <td>
        <span className={`${styles.trainingRunStatus} ${getStatusClass(run.status)}`}>{run.status}</span>
      </td>

      <td>{formatDate(run.created_at)}</td>

      <td>{run.finished_at ? formatDate(run.finished_at) : '—'}</td>

      <td>
        <span className={styles.trainingRunUser}>{run.user_id}</span>
      </td>
    </tr>
  )
}
