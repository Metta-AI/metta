/**
 * This file implements the search field and the search menu dropdown.
 * NOTE: The highlighting logic is done in the files that draw the objects.
 */

import { find, onEvent } from './htmlutils.js'
import { requestFrame } from './main.js'

/** The search state. */
export const search = {
  active: false,
  query: '',
  parts: [] as string[],
}

/** Update the search query and the search parts. */
onEvent('input', "#search-input", () => {
  let target = find('#search-input') as HTMLInputElement
  search.query = target.value
  search.active = search.query.length > 0
  search.parts = search.query.toLowerCase().split(' ').filter(part => part.length > 0)
  console.info('searchQuery:', search.query)
  requestFrame()
})

/** Check if the text matches the search query. */
export function searchMatch(text: string): boolean {
  for (const part of search.parts) {
    if (text.toLowerCase().includes(part)) {
      return true
    }
  }
  return false
}
