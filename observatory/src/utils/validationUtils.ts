/**
 * Validation utilities for dashboard state restoration
 */

// Cross-reference validation - only keep items that actually exist in available options
export const filterValidEvalNames = (
  savedEvalNames: string[],
  availableEvalNames: string[]
): { valid: string[]; invalid: string[] } => {
  const availableSet = new Set(availableEvalNames)
  const valid: string[] = []
  const invalid: string[] = []

  savedEvalNames.forEach((name) => {
    if (availableSet.has(name)) {
      valid.push(name)
    } else {
      invalid.push(name)
    }
  })

  return { valid, invalid }
}

export const filterValidIds = (savedIds: string[], availableIds: string[]): { valid: string[]; invalid: string[] } => {
  const availableSet = new Set(availableIds)
  const valid: string[] = []
  const invalid: string[] = []

  savedIds.forEach((id) => {
    if (availableSet.has(id)) {
      valid.push(id)
    } else {
      invalid.push(id)
    }
  })

  return { valid, invalid }
}

export const filterValidMetrics = (
  savedMetrics: string[],
  availableMetrics: string[]
): { valid: string[]; invalid: string[] } => {
  const availableSet = new Set(availableMetrics)
  const valid: string[] = []
  const invalid: string[] = []

  savedMetrics.forEach((metric) => {
    if (availableSet.has(metric)) {
      valid.push(metric)
    } else {
      invalid.push(metric)
    }
  })

  return { valid, invalid }
}
