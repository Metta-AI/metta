import React from 'react'
import styles from './MetricSelector.module.css'

interface MetricSelectorProps {
  metrics: string[]
  selectedMetric: string
  onSelectionChange: (metric: string) => void
  loading?: boolean
  disabled?: boolean
}

export const MetricSelector: React.FC<MetricSelectorProps> = ({
  metrics,
  selectedMetric,
  onSelectionChange,
  loading = false,
  disabled = false
}) => {
  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loadingContainer}>
          <span className={styles.loadingSpinner}></span>
          Loading metrics...
        </div>
      </div>
    )
  }

  if (disabled) {
    return (
      <div className={styles.container}>
        <div className={styles.disabledMessage}>
          Select policies and evaluations to see available metrics
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.selectContainer}>
        <select
          value={selectedMetric}
          onChange={(e) => onSelectionChange(e.target.value)}
          disabled={metrics.length === 0}
          className={styles.select}
        >
          <option value="">
            {metrics.length === 0 ? 'No metrics available' : 'Select a metric'}
          </option>
          {metrics.map(metric => (
            <option key={metric} value={metric}>
              {metric}
            </option>
          ))}
        </select>
      </div>

      <div className={`${styles.infoMessage} ${
        metrics.length > 0 ? styles.success : styles.warning
      }`}>
        {metrics.length > 0 ? (
          <>
            {metrics.length} metric{metrics.length !== 1 ? 's' : ''} available
            {selectedMetric && ` • ${selectedMetric} selected`}
          </>
        ) : (
          'No metrics found for the selected policies and evaluations'
        )}
      </div>
    </div>
  )
}