/**
 * This file implements the search field and the search menu dropdown.
 * NOTE: The highlighting logic is done in the files that draw the objects.
 */

import { find, findIn, onEvent, removeChildren, showDropdown } from './htmlutils.js'
import { requestFrame } from './main.js'
import { state } from './common.js'
import { propertyName, propertyIcon } from './replay.js'

var searchInput = find('#search-input') as HTMLInputElement
var searchDropdown = find('#search-dropdown')
searchDropdown.classList.add('hidden')
var searchItemTemplate = findIn(searchDropdown, '.search-item')
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
  let usedKeys = new Set<string>()
  let keys: string[] = []
  for (let key of state.replay.resource_inventory.keys()) {
    keys.push(key)
  }
  for (let key of state.replay.object_types) {
    keys.push(key)
  }

  for (let key of keys) {
    let searchItem = searchItemTemplate.cloneNode(true) as HTMLElement
    let shortKey = key.replace('inv:', '').replace('agent:', '')
    if (usedKeys.has(shortKey)) {
      continue
    }
    usedKeys.add(shortKey)
    searchItem.setAttribute('data-key', shortKey)
    searchItem.querySelector('.name')!.textContent = propertyName(key)
    let icon = searchItem.querySelector('.icon') as HTMLImageElement
    icon.src = propertyIcon(key)
    let filter = searchItem.querySelector('.filter') as HTMLImageElement
    if (search.parts.includes(shortKey)) {
      filter.src = 'data/ui/check-on.png'
    } else {
      filter.src = 'data/ui/check-off.png'
    }
    searchDropdown.appendChild(searchItem)
  }
}

/** Update the search query and the search parts. */
onEvent('input', '#search-input', () => {
  let target = find('#search-input') as HTMLInputElement
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

onEvent('click', '#search-input', (target: HTMLElement, event: Event) => {
  updateSearchDropdown()
  showDropdown(target, searchDropdown)
  requestFrame()
})

function remove(array: string[], item: string) {
  let index = array.indexOf(item)
  if (index != -1) {
    array.splice(index, 1)
  }
}

onEvent('click', '#search-dropdown .search-item', (target: HTMLElement, event: Event) => {
  let key = target.getAttribute('data-key')
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
