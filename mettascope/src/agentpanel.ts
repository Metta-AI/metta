import { state } from './common.js'
import {
  find,
  findAttr,
  finds,
  hideMenu,
  localStorageGetObject,
  localStorageSetObject,
  onEvent,
  removeChildren,
  showDropdown,
  showMenu,
} from './htmlutils.js'
import { updateSelection } from './main.js'
import { getAttr, propertyIcon, propertyName } from './replay.js'

enum SortDirection {
  None = 0,
  Descending = 1,
  Ascending = -1,
}

/** A column definition for the agent table. */
class ColumnDefinition {
  field: string
  isFinal: boolean
  sortDirection: SortDirection

  constructor(
    field: string,
    isFinal: boolean,
    sortDirection: SortDirection = SortDirection.None,
    _isStepColumn = false
  ) {
    this.field = field
    this.isFinal = isFinal
    this.sortDirection = sortDirection
  }

  generateName() {
    let name = propertyName(this.field)
    if (this.isFinal) {
      name = `Final: ${name}`
    }
    return name
  }

  generateIcon() {
    return propertyIcon(this.field)
  }

  // Unlike name, a tool tip has exact field.
  generateTooltip() {
    let tooltip = `${this.field} field`
    if (this.isFinal) {
      tooltip = `Final: ${tooltip}`
    }
    return tooltip
  }
}

const agentTable = find('#agent-panel .table')
const table = find('#agent-panel .table')
const columnTemplate = finds('#agent-panel .table .column')[0]
const headerCellTemplate = finds('#agent-panel .table .header-cell')[0]
const dataCellTemplate = finds('#agent-panel .table .data-cell')[0]
const newColumnTemplate = finds('#agent-panel .table .new-column')[0]
const newColumnHeaderCell = finds('#agent-panel .table .new-column .header-cell')[0]
const newColumnDataCell = finds('#agent-panel .table .new-column .data-cell')[0]
const columnMenu = find('#column-menu')
const newColumnDropdown = find('#new-column-dropdown')
const columnOptions = find('#new-column-dropdown .column-options')
const columnOptionTemplate = find('#new-column-dropdown .column-option')
const typeahead = find('#new-column-input') as HTMLInputElement

let columns = [
  new ColumnDefinition('agent_id', false),
  new ColumnDefinition('total_reward', false),
  new ColumnDefinition('total_reward', true),
]
let mainSort: ColumnDefinition = columns[1]
let typeaheadValue = ''

/** Swaps the element 1 position to the right. */
function swapRight(list: any[], element: any) {
  const index = list.indexOf(element)
  if (index === -1) {
    return
  }
  const tmp = list[index]
  list[index] = list[index + 1]
  list[index + 1] = tmp
}

/** Swaps the element 1 position to the left. */
function swapLeft(list: any[], element: any) {
  const index = list.indexOf(element)
  if (index === -1) {
    return
  }
  const tmp = list[index]
  list[index] = list[index - 1]
  list[index - 1] = tmp
}

/** Save the agent table to local storage. */
function saveAgentTable() {
  localStorageSetObject('agentPanelColumns', columns)
}

/** Initialize the agent table. */
export function initAgentTable() {
  // Load the columns from local storage.
  const plainColumns = localStorageGetObject('agentPanelColumns', columns)
  columns = plainColumns.map((column) => new ColumnDefinition(column.field, column.isFinal, column.sortDirection))

  // Hide the column menu and new column dropdown.
  columnMenu.classList.add('hidden')
  newColumnDropdown.classList.add('hidden')

  // Clear the templates for addition of elements.
  removeChildren(table)
  removeChildren(columnTemplate)
  removeChildren(newColumnTemplate)
}

/** Given an element, get the field and isFinal information thats up the DOM tree. */
function getFieldInfo(target: HTMLElement): {
  columnField: string
  columnIsFinal: boolean
} {
  const columnField = findAttr(target, 'data-column-field')
  const columnIsFinal = findAttr(target, 'data-column-is-final') === 'true'
  return { columnField, columnIsFinal }
}

/** Clicking on the column menu button should show the column menu. */
onEvent('click', '#agent-panel .header-cell .dropdown', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  const columnMenu = find('#column-menu')
  columnMenu.setAttribute('data-column-field', columnField)
  columnMenu.setAttribute('data-column-is-final', columnIsFinal.toString())
  showMenu(target, columnMenu)
})

/** Toggle the sort direction of the column. */
function toggleSortDirection(columnField: string, columnIsFinal: boolean) {
  columns.forEach((column) => {
    if (column.field === columnField && column.isFinal === columnIsFinal) {
      column.sortDirection = SortDirection.Ascending
      mainSort = column
    } else {
      column.sortDirection = SortDirection.None
    }
  })
  updateAgentTable()
  saveAgentTable()
  hideMenu()
}

/** Clicking on the sort up button should sort the column in ascending order. */
onEvent('click', '#column-menu .sort-up', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  toggleSortDirection(columnField, columnIsFinal)
})

/** Clicking on the sort down button should sort the column in descending order. */
onEvent('click', '#column-menu .sort-down', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  toggleSortDirection(columnField, columnIsFinal)
})

/** Clicking on the move left button should move the column to the left. */
onEvent('click', '#column-menu .move-left', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  const column = columns.find((column) => column.field === columnField && column.isFinal === columnIsFinal)
  if (column != null) {
    swapLeft(columns, column)
    updateAgentTable()
    saveAgentTable()
  }
  hideMenu()
})

/** Clicking on the move right button should move the column to the right. */
onEvent('click', '#column-menu .move-right', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  const column = columns.find((column) => column.field === columnField && column.isFinal === columnIsFinal)
  if (column != null) {
    swapRight(columns, column)
    updateAgentTable()
    saveAgentTable()
  }
  hideMenu()
})

/** Clicking on the hide column button should remove the column from the columns array. */
onEvent('click', '#column-menu .hide-column', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  columns = columns.filter((column) => !(column.field === columnField && column.isFinal === columnIsFinal))
  updateAgentTable()
  saveAgentTable()
  hideMenu()
})

/** Clicking on the table directly should set is as main sort column or cycle the sort direction. */
onEvent('click', '#agent-panel .header-cell', (target: HTMLElement, _e: Event) => {
  const { columnField, columnIsFinal } = getFieldInfo(target)
  if (columnField !== '') {
    for (const column of columns) {
      if (column.field === columnField && column.isFinal === columnIsFinal) {
        if (mainSort === column) {
          if (column.sortDirection === SortDirection.None) {
            column.sortDirection = SortDirection.Descending
          } else {
            column.sortDirection =
              column.sortDirection === SortDirection.Descending ? SortDirection.Ascending : SortDirection.Descending
          }
        } else {
          column.sortDirection = SortDirection.Descending
        }
        mainSort = column
      } else {
        column.sortDirection = SortDirection.None
      }
    }
    updateAgentTable()
    saveAgentTable()
  }
})

/** Clicking on a data cell should select the agent. */
onEvent('click', '#agent-panel .data-cell', (target: HTMLElement, _e: Event) => {
  const agentId = findAttr(target, 'data-agent-id')
  if (agentId !== '') {
    state.replay.objects.forEach((gridObject: any) => {
      if (gridObject.hasOwnProperty('agent_id') && getAttr(gridObject, 'agent_id') === agentId) {
        updateSelection(gridObject, true)
        // Note: can't use break with forEach
      }
    })
  }
})

/**
 * Clicking on the new column input should show the new column dropdown and
 * allow you to type-ahead to select or search for the column.
 */
onEvent('click', '#new-column-input', (target: HTMLElement, _e: Event) => {
  const newColumnDropdown = find('#new-column-dropdown')
  updateAvailableColumns()
  showDropdown(target, newColumnDropdown)
})

/** When the user types in the typeahead, filter the available columns. */
onEvent('input', '#new-column-input', (_target: HTMLElement, _e: Event) => {
  updateAvailableColumns()
})

/** Toggles the column in the columns array based on the field and isFinal. */
function toggleColumn(columnField: string, columnIsFinal: boolean) {
  let found = -1
  if (columnField !== '') {
    for (let i = 0; i < columns.length; i++) {
      if (columns[i].field === columnField && columns[i].isFinal === columnIsFinal) {
        found = i
      }
    }
  }
  if (found !== -1) {
    // Remove the column from the columns array.
    columns.splice(found, 1)
  } else {
    // Add the column to the columns array.
    columns.push(new ColumnDefinition(columnField, columnIsFinal))
  }
  updateAgentTable()
  updateAvailableColumns()
  saveAgentTable()
}

/**
 * Clicking on the step check should add or remove the "current step" column
 * from the columns array.
 */
onEvent('click', '#new-column-dropdown .step-check', (target: HTMLElement, _e: Event) => {
  toggleColumn(findAttr(target, 'data-column-field'), false)
})

/**
 * Clicking on the final check should add or remove the "final step" column
 * from the columns array.
 */
onEvent('click', '#new-column-dropdown .final-check', (target: HTMLElement, _e: Event) => {
  toggleColumn(findAttr(target, 'data-column-field'), true)
})

/** Update the available columns. */
export function updateAvailableColumns() {
  // The columns might change due to changes in:
  //   * The replay format.
  //   * The typeahead value.
  //   * The columns array.
  //   * The main sort column.

  const availableColumns: ColumnDefinition[] = []
  const typeahead = find('#new-column-input') as HTMLInputElement
  typeaheadValue = typeahead.value
  const noMatchFound = find('#new-column-dropdown .no-match-found')

  // All agent keys:
  const agentKeys = new Set<string>()
  for (const agent of state.replay.agents) {
    for (const key in agent) {
      agentKeys.add(key)
    }
  }
  // All inventory keys:
  for (const key of agentKeys) {
    if (key !== 'agent' && key !== 'c' && key !== 'r' && key !== 'reward') {
      // If there is a typeahead value, only show columns that match the typeahead value.
      if (typeaheadValue !== '' && !key.toLowerCase().includes(typeaheadValue.toLowerCase())) {
        continue
      }
      availableColumns.push(new ColumnDefinition(key, false))
    }
  }

  if (availableColumns.length === 0) {
    noMatchFound.classList.remove('hidden')
  } else {
    noMatchFound.classList.add('hidden')
  }

  removeChildren(columnOptions)

  for (const column of availableColumns) {
    const option = columnOptionTemplate.cloneNode(true) as HTMLElement
    option.querySelector('.name')!.textContent = column.generateName()
    option.querySelector('.icon')?.setAttribute('src', column.generateIcon())
    option.setAttribute('title', column.generateTooltip())
    option.setAttribute('data-column-field', column.field)
    let stepColumnExists = false
    let finalColumnExists = false
    for (const c of columns) {
      if (c.field === column.field && c.isFinal === false) {
        stepColumnExists = true
      } else if (c.field === column.field && c.isFinal === true) {
        finalColumnExists = true
      }
    }
    option
      .querySelector('.step-check')
      ?.setAttribute('src', stepColumnExists ? 'data/ui/check-on.png' : 'data/ui/check-off.png')
    option
      .querySelector('.final-check')
      ?.setAttribute('src', finalColumnExists ? 'data/ui/check-on.png' : 'data/ui/check-off.png')
    columnOptions.appendChild(option)
  }
}

/** Update the agent table. */
export function updateAgentTable() {
  // The agent table might change due to changes in:
  //   * The columns array.
  //   * The main sort column.
  //   * The sort direction.
  //   * The selected grid object.
  //   * The selected agent.

  removeChildren(agentTable)

  const list = state.replay.agents.slice()
  const agents = list.sort((a: any, b: any) => {
    let aValue: number
    let bValue: number
    if (mainSort.isFinal) {
      // Uses the final step for the sort.
      aValue = getAttr(a, mainSort.field, state.replay.max_steps - 1)
      bValue = getAttr(b, mainSort.field, state.replay.max_steps - 1)
    } else {
      // Uses the current step for the sort.
      aValue = getAttr(a, mainSort.field)
      bValue = getAttr(b, mainSort.field)
    }
    // Sort direction adjustment.
    if (mainSort.sortDirection === SortDirection.Descending) {
      return bValue - aValue
    }
    return aValue - bValue
  })

  // Create the columns.
  for (const columnDef of columns) {
    const column = columnTemplate.cloneNode(true) as HTMLElement
    column.setAttribute('data-column-field', columnDef.field)
    column.setAttribute('data-column-is-final', columnDef.isFinal.toString())
    const headerCell = headerCellTemplate.cloneNode(true) as HTMLElement
    const name = headerCell.querySelectorAll('.name')[0]
    name.textContent = columnDef.generateName()
    const icon = headerCell.querySelectorAll('.icon')[0]
    icon.setAttribute('src', columnDef.generateIcon())
    const sortIcon = headerCell.querySelector('.sort-icon') as HTMLElement
    if (columnDef.sortDirection === SortDirection.Descending) {
      sortIcon.setAttribute('src', 'data/ui/sort-down.png')
    } else if (columnDef.sortDirection === SortDirection.Ascending) {
      sortIcon.setAttribute('src', 'data/ui/sort-up.png')
    } else {
      sortIcon.classList.add('hidden')
    }
    headerCell.setAttribute('title', columnDef.generateTooltip())
    column.appendChild(headerCell)

    // Create the data cells.
    agents.forEach((agent: any) => {
      if (agent != null) {
        const dataCell = dataCellTemplate.cloneNode(true) as HTMLElement

        let value: number
        if (columnDef.isFinal) {
          value = getAttr(agent, columnDef.field, state.replay.max_steps - 1)
        } else {
          value = getAttr(agent, columnDef.field)
        }
        if (value == null) {
          value = 0
        }
        let valueStr = value.toString()
        if (valueStr.includes('.')) {
          valueStr = value.toFixed(3)
        }

        dataCell.children[0].textContent = valueStr
        const agentId = getAttr(agent, 'agent_id')
        dataCell.setAttribute('data-agent-id', agentId.toString())
        if (state.selectedGridObject != null && agentId === getAttr(state.selectedGridObject, 'agent_id')) {
          dataCell.classList.add('selected')
        }
        column.appendChild(dataCell)
      }
    })

    table.appendChild(column)
  }

  const newColumn = newColumnTemplate.cloneNode(true) as HTMLElement
  const headerCell = newColumnHeaderCell.cloneNode(true) as HTMLElement
  newColumn.appendChild(headerCell)
  agents.forEach((agent: any) => {
    const dataCell = newColumnDataCell.cloneNode(true) as HTMLElement
    const agentId = getAttr(agent, 'agent_id')
    dataCell.setAttribute('data-agent-id', agentId.toString())
    if (state.selectedGridObject != null && agentId === getAttr(state.selectedGridObject, 'agent_id')) {
      dataCell.classList.add('selected')
    }
    newColumn.appendChild(dataCell)
  })
  table.appendChild(newColumn)

  // Restore the typeahead value.
  typeahead.value = typeaheadValue
}
