import React from 'react'
import styles from './TrainingRunPolicySelector.module.css'

interface TrainingRunPolicySelectorProps {
  value: 'latest' | 'best'
  onChange: (value: 'latest' | 'best') => void
  disabled?: boolean
}

export const TrainingRunPolicySelector: React.FC<TrainingRunPolicySelectorProps> = ({
  value,
  onChange,
  disabled = false
}) => {
  if (disabled) {
    return (
      <div className={styles.container}>
        <div className={styles.disabledMessage}>
          Select policies to configure training run policy selection
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.radioGroup}>
        <div 
          className={`${styles.radioOption} ${value === 'latest' ? styles.selected : ''}`}
          onClick={() => onChange('latest')}
        >
          <input
            type="radio"
            id="latest"
            name="policy-selector"
            value="latest"
            checked={value === 'latest'}
            onChange={(e) => onChange(e.target.value as 'latest' | 'best')}
            className={styles.radioInput}
          />
          <div className={styles.radioContent}>
            <div className={styles.radioLabel}>Latest</div>
            <div className={styles.radioDescription}>
              Select the most recently trained policy from each training run (highest epoch).
            </div>
          </div>
        </div>

        <div 
          className={`${styles.radioOption} ${value === 'best' ? styles.selected : ''}`}
          onClick={() => onChange('best')}
        >
          <input
            type="radio"
            id="best"
            name="policy-selector"
            value="best"
            checked={value === 'best'}
            onChange={(e) => onChange(e.target.value as 'latest' | 'best')}
            className={styles.radioInput}
          />
          <div className={styles.radioContent}>
            <div className={styles.radioLabel}>Best</div>
            <div className={styles.radioDescription}>
              Select the best performing policy from each training run based on average score across all selected evaluations.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}