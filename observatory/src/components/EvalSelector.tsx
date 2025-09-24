import React, { useState } from 'react'
import styles from './EvalSelector.module.css'
import { getEvalCategory, groupEvalNamesByCategory, reconstructEvalName } from '../utils/evalNameUtils'

interface EvalSelectorProps {
  evalNames: Set<string>
  selectedEvalNames: Set<string>
  onSelectionChange: (evalNames: Set<string>) => void
  loading?: boolean
}

export const EvalSelector: React.FC<EvalSelectorProps> = ({
  evalNames,
  selectedEvalNames,
  onSelectionChange,
  loading = false,
}) => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())

  const categories = groupEvalNamesByCategory(evalNames)

  const toggleCategoryExpansion = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      return next
    })
  }

  const countSelectedEvalNamesInCategory = (category: string): number => {
    let count = 0
    for (const evalName of selectedEvalNames) {
      if (getEvalCategory(evalName) === category) {
        count++
      }
    }
    return count
  }

  const isCategorySelected = (category: string): boolean => {
    const envNames = categories.get(category)
    if (!envNames) {
      throw new Error(`No environment names found for category: ${category}`)
    }
    const count = countSelectedEvalNamesInCategory(category)
    return count === envNames.length
  }

  const isCategoryPartiallySelected = (category: string): boolean => {
    const envNames = categories.get(category)
    if (!envNames) {
      throw new Error(`No environment names found for category: ${category}`)
    }
    const count = countSelectedEvalNamesInCategory(category)
    return count > 0 && count < envNames.length
  }

  const isEnvSelected = (category: string, envName: string): boolean => {
    return selectedEvalNames.has(reconstructEvalName(category, envName))
  }

  const handleCategoryToggle = (category: string) => {
    const isCurrentlySelected = isCategorySelected(category)

    if (isCurrentlySelected) {
      // Deselect entire category
      onSelectionChange(new Set([...selectedEvalNames].filter((evalName) => getEvalCategory(evalName) !== category)))
    } else {
      // Select entire category
      const envNames = categories.get(category)
      if (!envNames) {
        throw new Error(`No environment names found for category: ${category}`)
      }
      const reconstructedNames = envNames.map((envName) => reconstructEvalName(category, envName))
      const newSelections = new Set([...selectedEvalNames, ...reconstructedNames])
      onSelectionChange(newSelections)
    }
  }

  const handleEnvToggle = (category: string, envName: string) => {
    const evalName = reconstructEvalName(category, envName)
    const newSelections = new Set(selectedEvalNames)
    if (newSelections.has(evalName)) {
      newSelections.delete(evalName)
    } else {
      newSelections.add(evalName)
    }
    onSelectionChange(newSelections)
  }

  const handleClearAll = () => {
    onSelectionChange(new Set())
  }

  const getTotalSelectedCount = () => {
    return selectedEvalNames.size
  }

  const getTotalAvailableCount = () => {
    return evalNames.size
  }

  return (
    <div className={styles.container}>
      <div className={styles.categoriesContainer}>
        {loading && (
          <div className={styles.loadingContainer}>
            <span className={styles.loadingSpinner}></span>
            Loading evaluations...
          </div>
        )}

        {!loading && evalNames.size === 0 && (
          <div className={styles.emptyMessage}>No evaluations available. Select policies first.</div>
        )}

        {!loading &&
          Array.from(categories.entries()).map(([category, envNames]) => (
            <div key={category} className={styles.categoryItem}>
              {/* Category Header */}
              <div className={styles.categoryHeader}>
                <div className={styles.categoryInfo}>
                  <input
                    type="checkbox"
                    checked={isCategorySelected(category)}
                    ref={(el) => {
                      if (el) {
                        el.indeterminate = isCategoryPartiallySelected(category)
                      }
                    }}
                    onChange={() => handleCategoryToggle(category)}
                    className={styles.categoryCheckbox}
                  />
                  <button onClick={() => toggleCategoryExpansion(category)} className={styles.categoryName}>
                    {category}
                  </button>
                  <span className={styles.envCount}>({envNames.length})</span>
                </div>
                <button
                  onClick={() => toggleCategoryExpansion(category)}
                  className={`${styles.expandIcon} ${expandedCategories.has(category) ? styles.expanded : ''}`}
                >
                  ▶
                </button>
              </div>

              {/* Environment List */}
              {expandedCategories.has(category) && (
                <div className={styles.environmentsList}>
                  {envNames.map((envName) => (
                    <div
                      key={envName}
                      className={`${styles.environmentItem} ${isEnvSelected(category, envName) ? styles.selected : ''}`}
                      onClick={() => handleEnvToggle(category, envName)}
                    >
                      <input
                        type="checkbox"
                        checked={isEnvSelected(category, envName)}
                        onChange={() => handleEnvToggle(category, envName)}
                        className={styles.environmentCheckbox}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <span className={styles.environmentName}>{envName}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
      </div>

      <div className={styles.selectedCount}>
        <span>
          {getTotalSelectedCount()} of {getTotalAvailableCount()} environments selected
        </span>
        <button onClick={handleClearAll} disabled={getTotalSelectedCount() === 0} className={styles.clearButton}>
          Clear All
        </button>
      </div>
    </div>
  )
}
