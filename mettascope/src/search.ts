/**
 * This file implements the search field and the search menu dropdown.
 * NOTE: The highlighting logic is done in the files that draw the objects.
 */

import { state } from './common.js'
import { find, findIn, onEvent, removeChildren, showDropdown } from './htmlutils.js'
import { requestFrame } from './main.js'
import { propertyIcon, propertyName } from './replay.js'

const searchInput = find('#search-input') as HTMLInputElement
const searchDropdown = find('#search-dropdown')
searchDropdown.classList.add('hidden')
const searchItemTemplate = findIn(searchDropdown, '.search-item')
searchItemTemplate.remove()

/** The search state. */
export const search = {
  active: false,
  query: '',
  parts: [] as string[],
}

function updateSearchDropdown() {
  removeChildren(searchDropdown)
  // Add all of the resources to the search dropdown.
  const usedKeys = new Set<string>()
  const keys: string[] = []
  for (const key of state.replay.itemNames) {
    keys.push(key)
  }
  for (const key of state.replay.typeNames) {
    keys.push(key)
  }

  for (const key of keys) {
    const searchItem = searchItemTemplate.cloneNode(true) as HTMLElement
    if (usedKeys.has(key)) {
      continue
    }
    usedKeys.add(key)
    searchItem.setAttribute('data-key', key)
    searchItem.querySelector('.name')!.textContent = propertyName(key)
    const icon = searchItem.querySelector('.icon') as HTMLImageElement
    icon.src = propertyIcon(key)
    const filter = searchItem.querySelector('.filter') as HTMLImageElement
    if (search.parts.includes(key)) {
      filter.src = 'data/ui/check-on.png'
    } else {
      filter.src = 'data/ui/check-off.png'
    }
    searchDropdown.appendChild(searchItem)
  }
}

/** Update the search query and the search parts. */
onEvent('input', '#search-input', () => {
  const target = find('#search-input') as HTMLInputElement
  search.query = target.value
  search.active = search.query.length > 0
  search.parts = search.query
    .toLowerCase()
    .split(' ')
    .filter((part) => part.length > 0)
  console.info('searchQuery:', search.query)
  updateSearchDropdown()
  requestFrame()
})

onEvent('click', '#search-input', (target: HTMLElement, _e: Event) => {
  updateSearchDropdown()
  showDropdown(target, searchDropdown)
  requestFrame()
})

function remove(array: string[], item: string) {
  const index = array.indexOf(item)
  if (index !== -1) {
    array.splice(index, 1)
  }
}

onEvent('click', '#search-dropdown .search-item', (target: HTMLElement, _e: Event) => {
  const key = target.getAttribute('data-key')
  if (key != null) {
    if (search.parts.includes(key)) {
      remove(search.parts, key)
    } else {
      search.parts.push(key)
    }
    search.query = search.parts.join(' ')
    searchInput.value = search.query
  }
  search.active = search.parts.length > 0
  updateSearchDropdown()
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
