/**
 * Utility functions for handling evaluation names and categories
 */

export const OVERALL_EVAL_NAME = 'overall'
export const DEFAULT_EVAL_CATEGORY = 'environments'

export const getShortName = (evalName: string): string => {
  if (evalName.toLowerCase() === OVERALL_EVAL_NAME) return evalName
  return evalName.split('/').pop() || evalName
}

export const getEvalCategory = (evalName: string): string => {
  const parts = evalName.split('/')
  return parts.length > 1 ? parts[0] : DEFAULT_EVAL_CATEGORY
}

export const getEnvName = (evalName: string): string => {
  const parts = evalName.split('/')
  return parts.length > 1 ? parts[1] : evalName
}

export const groupEvalNamesByCategory = (evalNames: string[] | Set<string>): Map<string, string[]> => {
  const categories = new Map<string, string[]>()
  const evalNamesArray = Array.isArray(evalNames) ? evalNames : Array.from(evalNames)

  evalNamesArray.forEach((evalName) => {
    const category = getEvalCategory(evalName)
    const envName = getEnvName(evalName)

    if (!categories.has(category)) {
      categories.set(category, [])
    }
    categories.get(category)!.push(envName)
  })

  return categories
}

export const reconstructEvalName = (category: string, envName: string): string => {
  return category === DEFAULT_EVAL_CATEGORY ? envName : `${category}/${envName}`
}
