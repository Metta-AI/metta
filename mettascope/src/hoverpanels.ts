// Info panels are used to display information about the current state of objects.

// Lower-level hover rules:
// * You need to hover over an Entity for one second for the info panel to show.
// * You can hover off the Entity, and the panel will stay visible for one second.
// * If you hover over the panel, it will stay visible as long as the mouse is over it.
// * If you drag the panel, it will detach and stay on the screen.
//   - It will only be closed by clicking on the X.
//   - It will lose its hover stem on the bottom when it's in detached mode.

// Hover panels show:
// * The properties of the Entity (like position, which is hidden).
// * The inventory of the Entity.
// * The recipe of the Entity.
// * The memory menu button.

import { Vec2f } from './vector_math.js'
import * as Common from './common.js'
import { ctx, ui, state, html, showToast } from './common.js'
import { findIn, removeChildren, parseHtmlColor, localStorageSetNumber, localStorageGetNumber } from './htmlutils.js'
import { find, onEvent } from './htmlutils.js'
import { Entity, propertyName, propertyIcon } from './replay.js'
import { updateStep } from './main.js'

/** An info panel. */
export class HoverPanel {
  public entity: any
  public div: HTMLElement

  constructor(entity: Entity) {
    this.entity = entity
    this.div = document.createElement('div')
  }

  public update() {
    updateDom(this.div, this.entity)
  }
}

onEvent('click', '.hover-panel .close', (target: HTMLElement, e: Event) => {
  let panel = target.parentElement as HTMLElement
  panel.remove()
  ui.hoverPanels = ui.hoverPanels.filter((p) => p.div !== panel)
})

let hoverPanelTemplate = find('.hover-panel') as HTMLElement
let paramTemplate = findIn(hoverPanelTemplate, '.param')
let itemTemplate = findIn(hoverPanelTemplate, '.inventory .item')
let recipeArrow = findIn(hoverPanelTemplate, '.recipe .arrow')
hoverPanelTemplate.remove()

let hoverPanel = hoverPanelTemplate.cloneNode(true) as HTMLElement
document.body.appendChild(hoverPanel)
findIn(hoverPanel, '.actions').classList.add('hidden')
hoverPanel.classList.add('hidden')

hoverPanel.addEventListener('mousedown', (e: MouseEvent) => {
  // Create a new info panel.
  if (ui.delayedHoverObject === null) return
  let panel = new HoverPanel(ui.delayedHoverObject)
  panel.div = hoverPanelTemplate.cloneNode(true) as HTMLElement
  panel.div.classList.add('draggable')
  let tip = findIn(panel.div, '.tip')
  tip.remove()
  document.body.appendChild(panel.div)
  updateDom(panel.div, panel.entity)
  panel.div.style.top = hoverPanel.style.top
  panel.div.style.left = hoverPanel.style.left

  // Show the actions buttons (memory, etc.) if the Entity is an agent
  // and if the websocket is connected.
  let actions = findIn(panel.div, '.actions')
  if (state.ws != null && panel.entity.hasOwnProperty('agent_id')) {
    actions.classList.remove('hidden')
  } else {
    actions.classList.add('hidden')
  }

  ui.dragHtml = panel.div
  // Compute mouse position relative to the panel.
  let rect = panel.div.getBoundingClientRect()
  ui.dragOffset = new Vec2f(e.clientX - rect.left, e.clientY - rect.top)
  ui.dragging = 'info-panel'
  ui.hoverPanels.push(panel)

  // Hide the old hover panel.
  // The new info panel should be identical to the old hover panel,
  // so that the user sees no difference.
  hoverPanel.classList.add('hidden')
  ui.hoverObject = null
  ui.delayedHoverObject = null
  e.stopPropagation()
})

/** Updates the hover panel's visibility, position, and DOM tree. */
export function updateHoverPanel(entity: Entity | null) {
  if (!state.replay) return

  if (entity === null) {
    hoverPanel.classList.add('hidden')
    return
  }

  const typeId = entity.typeId
  let typeName = state.replay.typeNames[typeId as number]
  if (typeName == 'wall') {
    // Don't show hover panel for walls.
    hoverPanel.classList.add('hidden')
    return
  }

  updateDom(hoverPanel, entity)
  hoverPanel.classList.remove('hidden')

  let panelRect = hoverPanel.getBoundingClientRect()

  let position = entity.position.get()
  if (!position) return
  let x = position[0] * Common.TILE_SIZE
  let y = position[1] * Common.TILE_SIZE

  let uiPoint = ui.mapPanel.transformInner(new Vec2f(x, y - Common.TILE_SIZE / 2))

  // Put it in the center above the Entity.
  hoverPanel.style.left = uiPoint.x() - panelRect.width / 2 + 'px'
  hoverPanel.style.top = uiPoint.y() - panelRect.height + 'px'
}

/** Hides the hover panel. */
export function hideHoverPanel() {
  ui.delayedHoverObject = null
  hoverPanel.classList.add('hidden')
}

/** Updates the DOM tree of the info panel. */
function updateDom(htmlPanel: HTMLElement, entity: Entity) {
  // Update the readout.
  htmlPanel.setAttribute('data-entity-id', (entity.id || 0).toString())
  const agentId = entity.agentId
  htmlPanel.setAttribute('data-agent-id', (agentId || -1).toString())

  let params = findIn(htmlPanel, '.params')
  removeChildren(params)
  let inventory = findIn(htmlPanel, '.inventory')
  removeChildren(inventory)

  // Update entity properties display
  let param = paramTemplate.cloneNode(true) as HTMLElement

  // Show entity ID
  param.querySelector('.name')!.textContent = 'ID'
  param.querySelector('.value')!.textContent = entity.id.toString()
  params.appendChild(param)

  // Show entity type
  if (state.replay && entity.typeId) {
    const typeId = entity.typeId
    if (typeId != null && typeId < state.replay.typeNames.length) {
      param = paramTemplate.cloneNode(true) as HTMLElement
      param.querySelector('.name')!.textContent = 'Type'
      param.querySelector('.value')!.textContent = state.replay.typeNames[typeId]
      params.appendChild(param)
    }
  }

  // Show agent ID with color name (reuse existing agentId variable)
  if (agentId != null && agentId != 0) {
    param = paramTemplate.cloneNode(true) as HTMLElement
    param.querySelector('.name')!.textContent = 'Agent'
    let agentText = agentId.toString()
    param.querySelector('.value')!.textContent = agentText
    params.appendChild(param)
  }

  // Show total reward if available
  const totalReward = entity.totalReward?.get()
  if (totalReward != null && totalReward != 0) {
    param = paramTemplate.cloneNode(true) as HTMLElement
    param.querySelector('.name')!.textContent = 'Total Reward'
    const rewardText = typeof totalReward === 'number' && !Number.isInteger(totalReward)
      ? totalReward.toFixed(3)
      : totalReward.toString()
    param.querySelector('.value')!.textContent = rewardText
    params.appendChild(param)
  }

  // Update inventory display
  let map = new Map<string, number>()
  for (let itemId in entity.inventory.get()) {
    const itemName = state.replay!.itemNames[itemId]
    if (map.has(itemName)) {
      map.set(itemName, map.get(itemName)! + 1)
    } else {
      map.set(itemName, 1)
    }
  }
  for (let [itemName, quantity] of map.entries()) {
    let item = itemTemplate.cloneNode(true) as HTMLElement
    item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + itemName + '.png')
    item.querySelector('.amount')!.textContent = quantity.toString()
    inventory.appendChild(item)
  }

  // Update recipe display
  let recipe = findIn(htmlPanel, '.recipe')
  removeChildren(recipe)
  let recipeArea = findIn(htmlPanel, '.recipe-area')

  const recipeInput = entity.recipeInput
  const recipeOutput = entity.recipeOutput

  // Create maps for input and output items
  let inputMap = new Map<string, number>()
  let outputMap = new Map<string, number>()

  // Build input map
  if (recipeInput != null && Array.isArray(recipeInput) && state.replay) {
    for (let itemId of recipeInput) {
      if (itemId > 0 && itemId < state.replay.itemNames.length) {
        const itemName = state.replay.itemNames[itemId]
        inputMap.set(itemName, (inputMap.get(itemName) || 0) + 1)
      }
    }
  }

  // Build output map
  if (recipeOutput != null && Array.isArray(recipeOutput) && state.replay) {
    for (let itemId of recipeOutput) {
      if (itemId > 0 && itemId < state.replay.itemNames.length) {
        const itemName = state.replay.itemNames[itemId]
        outputMap.set(itemName, (outputMap.get(itemName) || 0) + 1)
      }
    }
  }

  // Display recipe if there are any inputs or outputs
  if (inputMap.size > 0 || outputMap.size > 0) {
    recipeArea.classList.remove('hidden')

    // Show input items
    for (let [itemName, quantity] of inputMap.entries()) {
      const inputItem = itemTemplate.cloneNode(true) as HTMLElement
      const inputIcon = inputItem.querySelector('.icon') as HTMLImageElement
      if (inputIcon) {
        inputIcon.src = 'data/atlas/resources/' + itemName + '.png'
      }
      const inputAmount = inputItem.querySelector('.amount')
      if (inputAmount) {
        inputAmount.textContent = quantity.toString()
      }
      recipe.appendChild(inputItem)
    }

    // Add arrow between inputs and outputs
    if (inputMap.size > 0 && outputMap.size > 0 && recipeArrow) {
      recipe.appendChild(recipeArrow.cloneNode(true))
    }

    // Show output items
    for (let [itemName, quantity] of outputMap.entries()) {
      const outputItem = itemTemplate.cloneNode(true) as HTMLElement
      const outputIcon = outputItem.querySelector('.icon') as HTMLImageElement
      if (outputIcon) {
        outputIcon.src = 'data/atlas/resources/' + itemName + '.png'
      }
      const outputAmount = outputItem.querySelector('.amount')
      if (outputAmount) {
        outputAmount.textContent = quantity.toString()
      }
      recipe.appendChild(outputItem)
    }
  } else {
    recipeArea.classList.add('hidden')
  }
}

/** Updates the readout of the selected Entity or replay info. */
export function updateReadout() {
  if (!state.replay) return

  let readout = ''
  readout += 'Step: ' + state.step + '\n'
  readout += 'Map size: ' + state.replay.mapSize[0] + 'x' + state.replay.mapSize[1] + '\n'
  readout += 'Num agents: ' + state.replay.numAgents + '\n'
  readout += 'Max steps: ' + state.replay.maxSteps + '\n'

  let objectTypeCounts = new Map<string, number>()
  for (const obj of state.replay.objects) {
    const type = obj.typeId
    const typeName = state.replay.typeNames[type as number]
    objectTypeCounts.set(typeName, (objectTypeCounts.get(typeName) || 0) + 1)
  }
  for (const [key, value] of objectTypeCounts.entries()) {
    readout += key + ' count: ' + value + '\n'
  }
  let info = find('#info-panel .info')
  if (info !== null) {
    info.innerHTML = readout
  }
}
