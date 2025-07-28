// Info panels are used to display information about the current state of objects.

// Lower-level hover rules:
// * You need to hover over an object for one second for the info panel to show.
// * You can hover off the object, and the panel will stay visible for one second.
// * If you hover over the panel, it will stay visible as long as the mouse is over it.
// * If you drag the panel, it will detach and stay on the screen.
//   - It will only be closed by clicking on the X.
//   - It will lose its hover stem on the bottom when it's in detached mode.

// Hover panels show:
// * The properties of the object (like position, which is hidden).
// * The inventory of the object.
// * The recipe of the object.
// * The memory menu button.

import { find, findIn, onEvent, removeChildren, findAttr } from './htmlutils.js'
import { state, ui } from './common.js'
import { getAttr, getObjectConfig } from './replay.js'
import * as Common from './common.js'
import { Vec2f } from './vector_math.js'

/** An info panel. */
export class HoverPanel {
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
  let panel = new HoverPanel(ui.delayedHoverObject)
  panel.div = hoverPanelTemplate.cloneNode(true) as HTMLElement
  panel.div.classList.add('draggable')
  let tip = findIn(panel.div, '.tip')
  tip.remove()
  document.body.appendChild(panel.div)
  updateDom(panel.div, panel.object)
  panel.div.style.top = hoverPanel.style.top
  panel.div.style.left = hoverPanel.style.left

  // Show the actions buttons (memory, etc.) if the object is an agent
  // and if the websocket is connected.
  let actions = findIn(panel.div, '.actions')
  if (state.ws != null && panel.object.hasOwnProperty('agent_id')) {
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
export function updateHoverPanel(object: any) {
  if (object !== null && object !== undefined) {
    let typeName = state.replay.object_types[getAttr(object, 'type')]
    if (typeName == 'wall') {
      // Don't show hover panel for walls.
      hoverPanel.classList.add('hidden')
      return
    }

    updateDom(hoverPanel, object)
    hoverPanel.classList.remove('hidden')

    let panelRect = hoverPanel.getBoundingClientRect()

    let x = getAttr(object, 'c') * Common.TILE_SIZE
    let y = getAttr(object, 'r') * Common.TILE_SIZE

    let uiPoint = ui.mapPanel.transformInner(new Vec2f(x, y - Common.TILE_SIZE / 2))

    // Put it in the center above the object.
    hoverPanel.style.left = uiPoint.x() - panelRect.width / 2 + 'px'
    hoverPanel.style.top = uiPoint.y() - panelRect.height + 'px'
  } else {
    hoverPanel.classList.add('hidden')
  }
  findIn(hoverPanel, '.close').classList.add('hidden')
}

/** Hides the hover panel. */
export function hideHoverPanel() {
  ui.delayedHoverObject = null
  hoverPanel.classList.add('hidden')
}

/** Updates the DOM tree of the info panel. */
function updateDom(htmlPanel: HTMLElement, object: any) {
  // Update the readout.
  htmlPanel.setAttribute('data-object-id', getAttr(object, 'id'))
  htmlPanel.setAttribute('data-agent-id', getAttr(object, 'agent_id'))

  let params = findIn(htmlPanel, '.params')
  removeChildren(params)
  let inventory = findIn(htmlPanel, '.inventory')
  removeChildren(inventory)
  for (const key in object) {
    let value = getAttr(object, key)
    if ((key.startsWith('inv:') || key.startsWith('agent:inv:')) && value > 0) {
      let item = itemTemplate.cloneNode(true) as HTMLElement
      item.querySelector('.amount')!.textContent = value
      let resource = key.replace('inv:', '').replace('agent:', '')
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
      let param = paramTemplate.cloneNode(true) as HTMLElement
      param.querySelector('.name')!.textContent = key
      param.querySelector('.value')!.textContent = value
      params.appendChild(param)
    }
  }

  // Populate the recipe area if the object config has input_ or output_ resources.
  let recipe = findIn(htmlPanel, '.recipe')
  removeChildren(recipe)
  let recipeArea = findIn(htmlPanel, '.recipe-area')
  let objectConfig = getObjectConfig(object)
  let displayedResources = 0
  if (objectConfig != null) {
    recipeArea.classList.remove('hidden')

    // If config has input_resources or output_resources use that,
    // otherwise use input_{resource} and output_{resource}.
    if (objectConfig.hasOwnProperty('input_resources') || objectConfig.hasOwnProperty('output_resources')) {
      // input_resources is a object like {heart: 1, blueprint: 1}
      for (let resource in objectConfig.input_resources) {
        let item = itemTemplate.cloneNode(true) as HTMLElement
        item.querySelector('.amount')!.textContent = objectConfig.input_resources[resource]
        item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
        recipe.appendChild(item)
        displayedResources++
      }
      // Add the arrow.
      recipe.appendChild(recipeArrow.cloneNode(true))
      // Add the output.
      if (objectConfig.hasOwnProperty('output_resources')) {
        for (let resource in objectConfig.output_resources) {
          let item = itemTemplate.cloneNode(true) as HTMLElement
          item.querySelector('.amount')!.textContent = objectConfig.output_resources[resource]
          item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
          recipe.appendChild(item)
          displayedResources++
        }
      }
    } else {
      // Configs have input_{resource} and output_{resource}.
      for (let key in objectConfig) {
        if (key.startsWith('input_')) {
          let resource = key.replace('input_', '')
          let amount = objectConfig[key]
          let item = itemTemplate.cloneNode(true) as HTMLElement
          item.querySelector('.amount')!.textContent = amount
          item.querySelector('.icon')!.setAttribute('src', 'data/atlas/resources/' + resource + '.png')
          recipe.appendChild(item)
          displayedResources++
        }
      }
      // Add the arrow.
      recipe.appendChild(recipeArrow.cloneNode(true))
      // Add the output.
      for (let key in objectConfig) {
        if (key.startsWith('output_')) {
          let resource = key.replace('output_', '')
          let amount = objectConfig[key]
          let item = itemTemplate.cloneNode(true) as HTMLElement
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

  let objectTypeCounts = new Map<string, number>()
  for (const gridObject of state.replay.grid_objects) {
    const type = getAttr(gridObject, 'type')
    const typeName = state.replay.object_types[type]
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
