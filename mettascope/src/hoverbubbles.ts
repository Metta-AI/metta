// Info bubbles are used to display information about the current state of objects.

// Lower-level hover rules:
// * You need to hover over an object for one second for the info bubble to show.
// * You can hover off the object, and the bubble will stay visible for one second.
// * If you hover over the bubble, it will stay visible as long as the mouse is over it.
// * If you drag the bubble, it will detach and stay on the screen.
//   - It will only be closed by clicking on the X.
//   - It will lose its hover stem on the bottom when it's in detached mode.

// Hover bubbles show:
// * The properties of the object (like position, which is hidden).
// * The inventory of the object.
// * The recipe of the object.
// * The memory menu button.

import * as Common from './common.js'
import { state, ui } from './common.js'
import { find, findIn, onEvent, removeChildren } from './htmlutils.js'
import { Vec2f } from './vector_math.js'
import { Entity } from './replay.js'

/** An info bubble. */
export class HoverBubble {
  public object: Entity
  public div: HTMLElement

  constructor(object: Entity) {
    this.object = object
    this.div = document.createElement('div')
  }

  public update() {
    updateDom(this.div, this.object)
  }
}

onEvent('click', '.hover-panel .close', (target: HTMLElement, _e: Event) => {
  const bubble = target.parentElement as HTMLElement
  bubble.remove()
  ui.hoverBubbles = ui.hoverBubbles.filter((p) => p.div !== bubble)
})

const hoverBubbleTemplate = find('.hover-panel') as HTMLElement
const paramTemplate = findIn(hoverBubbleTemplate, '.param')
const itemTemplate = findIn(hoverBubbleTemplate, '.inventory .item')
const recipeArrow = findIn(hoverBubbleTemplate, '.recipe .arrow')
hoverBubbleTemplate.remove()

const hoverBubble = hoverBubbleTemplate.cloneNode(true) as HTMLElement
document.body.appendChild(hoverBubble)
findIn(hoverBubble, '.actions').classList.add('hidden')
hoverBubble.classList.add('hidden')

hoverBubble.addEventListener('pointerdown', (e: PointerEvent) => {
  // Create a new info bubble.
  const bubble = new HoverBubble(ui.delayedHoverObject)
  bubble.div = hoverBubbleTemplate.cloneNode(true) as HTMLElement
  bubble.div.classList.add('draggable')
  const tip = findIn(bubble.div, '.tip')
  tip.remove()
  document.body.appendChild(bubble.div)
  updateDom(bubble.div, bubble.object)
  bubble.div.style.top = hoverBubble.style.top
  bubble.div.style.left = hoverBubble.style.left

  // Show the actions buttons (memory, etc.) if the object is an agent
  // and if the websocket is connected.
  const actions = findIn(bubble.div, '.actions')
  if (state.ws != null && bubble.object.isAgent) {
    actions.classList.remove('hidden')
  } else {
    actions.classList.add('hidden')
  }

  ui.dragHtml = bubble.div
  // Compute mouse position relative to the bubble.
  const rect = bubble.div.getBoundingClientRect()
  ui.dragOffset = new Vec2f(e.clientX - rect.left, e.clientY - rect.top)
  ui.dragging = 'info-bubble'
  ui.hoverBubbles.push(bubble)

  // Hide the old hover bubble.
  // The new info bubble should be identical to the old hover bubble,
  // so that the user sees no difference.
  hoverBubble.classList.add('hidden')
  ui.hoverObject = null
  ui.delayedHoverObject = null
  e.stopPropagation()
})

/** Updates the hover bubble's visibility, position, and DOM tree. */
export function updateHoverBubble(object: Entity) {
  if (object !== null && object !== undefined) {
    // Is there a popup open for this object?
    // Then don't show a new one.
    for (const bubble of ui.hoverBubbles) {
      if (bubble.object === object) {
        return
      }
    }

    const typeName = state.replay.typeNames[object.typeId]
    if (typeName === 'wall') {
      // Don't show hover bubble for walls.
      hoverBubble.classList.add('hidden')
      return
    }

    updateDom(hoverBubble, object)
    hoverBubble.classList.remove('hidden')

    const bubbleRect = hoverBubble.getBoundingClientRect()

    const location = object.location.get()
    const x = location[0] * Common.TILE_SIZE
    const y = location[1] * Common.TILE_SIZE

    const uiPoint = ui.mapPanel.transformInner(new Vec2f(x, y - Common.TILE_SIZE / 2))

    // Put it in the center above the object.
    hoverBubble.style.left = `${uiPoint.x() - bubbleRect.width / 2}px`
    hoverBubble.style.top = `${uiPoint.y() - bubbleRect.height}px`
  } else {
    hoverBubble.classList.add('hidden')
  }
  findIn(hoverBubble, '.close').classList.add('hidden')
}

/** Hides the hover bubble. */
export function hideHoverBubble() {
  ui.delayedHoverObject = null
  hoverBubble.classList.add('hidden')
}

/** Updates the DOM tree of the info bubble. */
function updateDom(htmlBubble: HTMLElement, object: Entity) {
  // Update the readout.
  htmlBubble.setAttribute('data-object-id', object.id.toString())
  htmlBubble.setAttribute('data-agent-id', object.agentId.toString())

  const params = findIn(htmlBubble, '.params')
  removeChildren(params)

  function addParam(name: string, value: string) {
    const param = paramTemplate.cloneNode(true) as HTMLElement
    param.querySelector('.name')!.textContent = name
    param.querySelector('.value')!.textContent = value
    params.appendChild(param)
  }

  // Add various parameters.
  addParam("ID", object.id.toString())
  const typeName = state.replay.typeNames[object.typeId]
  addParam('Type', typeName)
  if (object.isAgent) {
    addParam('Agent ID', object.agentId.toString())
    addParam('Current Reward', object.currentReward.get().toString())
  }

  // Populate the inventory area.
  const inventory = findIn(htmlBubble, '.inventory')
  removeChildren(inventory)
  for (const inventoryPair of object.inventory.get()) {
    const inventoryId = inventoryPair[0]
    const resourceAmount = inventoryPair[1]
    if (resourceAmount > 0) {
      const resourceName = state.replay.itemNames[inventoryId]
      const item = itemTemplate.cloneNode(true) as HTMLElement
      item.querySelector('.amount')!.textContent = resourceAmount.toString()
      item.querySelector('.icon')?.setAttribute('src', `data/atlas/resources/${resourceName}.png`)
      inventory.appendChild(item)
    }
  }

  // Populate the recipe area if the object config has input_ or output_ resources.
  const recipe = findIn(htmlBubble, '.recipe')
  removeChildren(recipe)
  const recipeArea = findIn(htmlBubble, '.recipe-area')

  let displayedResources = 0

  const inputResources = object.inputResources
  const outputResources = object.outputResources

  if (inputResources.length > 0 || outputResources.length > 0) {
    // Add the input resources.
    for (const resourcePair of inputResources) {
      const resourceId = resourcePair[0]
      const resourceAmount = resourcePair[1]
      const resourceName = state.replay.itemNames[resourceId]
      const item = itemTemplate.cloneNode(true) as HTMLElement
      item.querySelector('.amount')!.textContent = resourceAmount.toString()
      item.querySelector('.icon')?.setAttribute('src', `data/atlas/resources/${resourceName}.png`)
      recipe.appendChild(item)
      displayedResources++
    }
    // Add the arrow between the input and output.
    recipe.appendChild(recipeArrow.cloneNode(true))
    // Add the output resources.
    if (outputResources.length > 0) {
      for (const resourcePair of outputResources) {
        const resourceId = resourcePair[0]
        const resourceAmount = resourcePair[1]
        const resourceName = state.replay.itemNames[resourceId]
        const item = itemTemplate.cloneNode(true) as HTMLElement
        item.querySelector('.amount')!.textContent = resourceAmount.toString()
        item.querySelector('.icon')?.setAttribute('src', `data/atlas/resources/${resourceName}.png`)
        recipe.appendChild(item)
        displayedResources++
      }
    }
  }

  if (displayedResources > 0) {
    recipeArea.classList.remove('hidden')
  } else {
    recipeArea.classList.add('hidden')
  }
}

/** Updates the readout of the selected object or replay info. */
export function updateReadout() {
  let readout = ''
  readout += `Step: ${state.step}\n`
  readout += `Map size: ${state.replay.mapSize[0]}x${state.replay.mapSize[1]}\n`
  readout += `Num agents: ${state.replay.numAgents}\n`
  readout += `Max steps: ${state.replay.maxSteps}\n`

  const objectTypeCounts = new Map<string, number>()
  for (const gridObject of state.replay.objects) {
    const typeId = gridObject.typeId
    const typeName = state.replay.typeNames[typeId]
    objectTypeCounts.set(typeName, (objectTypeCounts.get(typeName) || 0) + 1)
  }
  for (const [key, value] of objectTypeCounts.entries()) {
    readout += `${key} count: ${value}\n`
  }
  const info = find('#info-panel .info')
  if (info !== null) {
    info.innerHTML = readout
  }
}
