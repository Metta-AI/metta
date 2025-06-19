/**
 * This file implements the search field and the search menu dropdown.
 */

import { find, onEvent } from './htmlutils.js'
import { requestFrame } from './worldmap.js'

export const search = {
  active: false,
  query: '',
  parts: [] as string[],
}

onEvent('input', "#search-input", () => {
  let target = find('#search-input') as HTMLInputElement
  search.query = target.value
  search.active = search.query.length > 0
  search.parts = search.query.toLowerCase().split(' ').filter(part => part.length > 0)
  console.info('searchQuery:', search.query)
  requestFrame()
})

export function searchMatch(text: string): boolean {
  for (const part of search.parts) {
    if (text.toLowerCase().includes(part)) {
      return true
    }
  }
  return false
}
