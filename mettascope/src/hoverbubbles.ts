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

import { state, ui } from './common.js'
import * as Common from './common.js'
import { find, findAttr, findIn, onEvent, removeChildren } from './htmlutils.js'
import { getAttr, getObjectConfig } from './replay.js'
import { Vec2f } from './vector_math.js'

/** An info bubble. */
export class HoverBubble {
  public object: any
  public div: HTMLElement

  constructor(object: any) {
    this.object = object
    this.div = document.createElement('div')
  }

  public update() {
    updateDom(this.div, this.object)
  }
}

onEvent('click', '.hover-panel .close', (target: HTMLElement, e: Event) => {
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
  if (state.ws != null && bubble.object.hasOwnProperty('agent_id')) {
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
export function updateHoverBubble(object: any) {
  if (object !== null && object !== undefined) {
    // Is there a popup open for this object?
    // Then don't show a new one.
    for (const bubble of ui.hoverBubbles) {
      if (bubble.object === object) {
        return
      }
    }

    const typeName = state.replay.object_types[getAttr(object, 'type')]
    if (typeName == 'wall') {
      // Don't show hover bubble for walls.
      hoverBubble.classList.add('hidden')
      return
    }

    updateDom(hoverBubble, object)
    hoverBubble.classList.remove('hidden')

    const bubbleRect = hoverBubble.getBoundingClientRect()

    const x = getAttr(object, 'c') * Common.TILE_SIZE
    const y = getAttr(object, 'r') * Common.TILE_SIZE

    const uiPoint = ui.mapPanel.transformInner(new Vec2f(x, y - Common.TILE_SIZE / 2))

    // Put it in the center above the object.
    hoverBubble.style.left = uiPoint.x() - bubbleRect.width / 2 + 'px'
    hoverBubble.style.top = uiPoint.y() - bubbleRect.height + 'px'
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
function updateDom(htmlBubble: HTMLElement, object: any) {
  // Update the readout.
  htmlBubble.setAttribute('data-object-id', getAttr(object, 'id'))
  htmlBubble.setAttribute('data-agent-id', getAttr(object, 'agent_id'))

  const params = findIn(htmlBubble, '.params')
  removeChildren(params)
  const inventory = findIn(htmlBubble, '.inventory')
  removeChildren(inventory)
  for (const key in object) {
    let value = getAttr(object, key)
    if ((key.startsWith('inv:') || key.startsWith('agent:inv:')) && value > 0) {
      const item = itemTemplate.cloneNode(true) as HTMLElement
      item.querySelector('.amount')!.textContent = value
      const resource = key.replace('inv:', '').replace('agent:', '')
      item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
      inventory.appendChild(item)
    } else {
      if (key == 'type') {
        value = state.replay.object_types[value]
      } else if (key == 'agent:color' && value >= 0 && value < Common.COLORS.length) {
        value = Common.COLORS[value][0]
      } else if (['group', 'total_reward', 'agent_id'].includes(key)) {
        // If the value is a float and not an integer, round it to three decimal places.
        if (typeof value === 'number' && !Number.isInteger(value)) {
          value = value.toFixed(3)
        }
      } else {
        continue
      }
      const param = paramTemplate.cloneNode(true) as HTMLElement
      param.querySelector('.name')!.textContent = key
      param.querySelector('.value')!.textContent = value
      params.appendChild(param)
    }
  }

  // Populate the recipe area if the object config has input_ or output_ resources.
  const recipe = findIn(htmlBubble, '.recipe')
  removeChildren(recipe)
  const recipeArea = findIn(htmlBubble, '.recipe-area')
  const objectConfig = getObjectConfig(object)
  let displayedResources = 0
  if (objectConfig != null) {
    recipeArea.classList.remove('hidden')

    // If config has input_resources or output_resources use that,
    // otherwise use input_{resource} and output_{resource}.
    if (objectConfig.hasOwnProperty('input_resources') || objectConfig.hasOwnProperty('output_resources')) {
      // input_resources is a object like {heart: 1, blueprint: 1}
      for (const resource in objectConfig.input_resources) {
        const item = itemTemplate.cloneNode(true) as HTMLElement
        item.querySelector('.amount')!.textContent = objectConfig.input_resources[resource]
        item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
        recipe.appendChild(item)
        displayedResources++
      }
      // Add the arrow.
      recipe.appendChild(recipeArrow.cloneNode(true))
      // Add the output.
      if (objectConfig.hasOwnProperty('output_resources')) {
        for (const resource in objectConfig.output_resources) {
          const item = itemTemplate.cloneNode(true) as HTMLElement
          item.querySelector('.amount')!.textContent = objectConfig.output_resources[resource]
          item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
          recipe.appendChild(item)
          displayedResources++
        }
      }
    } else {
      // Configs have input_{resource} and output_{resource}.
      for (const key in objectConfig) {
        if (key.startsWith('input_')) {
          const resource = key.replace('input_', '')
          const amount = objectConfig[key]
          const item = itemTemplate.cloneNode(true) as HTMLElement
          item.querySelector('.amount')!.textContent = amount
          item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
          recipe.appendChild(item)
          displayedResources++
        }
      }
      // Add the arrow.
      recipe.appendChild(recipeArrow.cloneNode(true))
      // Add the output.
      for (const key in objectConfig) {
        if (key.startsWith('output_')) {
          const resource = key.replace('output_', '')
          const amount = objectConfig[key]
          const item = itemTemplate.cloneNode(true) as HTMLElement
          item.querySelector('.amount')!.textContent = amount
          item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
          recipe.appendChild(item)
          displayedResources++
        }
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
  readout += 'Step: ' + state.step + '\n'
  readout += 'Map size: ' + state.replay.map_size[0] + 'x' + state.replay.map_size[1] + '\n'
  readout += 'Num agents: ' + state.replay.num_agents + '\n'
  readout += 'Max steps: ' + state.replay.max_steps + '\n'

  const objectTypeCounts = new Map<string, number>()
  for (const gridObject of state.replay.grid_objects) {
    const type = getAttr(gridObject, 'type')
    const typeName = state.replay.object_types[type]
    objectTypeCounts.set(typeName, (objectTypeCounts.get(typeName) || 0) + 1)
  }
  for (const [key, value] of objectTypeCounts.entries()) {
    readout += key + ' count: ' + value + '\n'
  }
  const info = find('#info-panel .info')
  if (info !== null) {
    info.innerHTML = readout
  }
}
