/**
 * Custom layout system for managing tabs and panes.
 */

// Mock panel types for testing.
export enum PanelType {
  LOGS = 'Logs',
  METRICS = 'Metrics',
  MAP_VIEW = 'Map View',
  AGENT_DETAILS = 'Agent Details',
}

export enum LayoutDirection {
  HORIZONTAL = 'horizontal',
  VERTICAL = 'vertical',
}

export enum DropZone {
  TAB_BAR = 'tab-bar',
  LEFT = 'left',
  RIGHT = 'right',
  TOP = 'top',
  BOTTOM = 'bottom',
}

/** A single tab and it's content for a pane. */
export class Tab {
  public title: string
  public content: string
  public isActive: boolean = false
  public panelType: PanelType

  constructor(title: string, content: string, panelType: PanelType = PanelType.LOGS) {
    this.title = title
    this.content = content
    this.panelType = panelType
  }
}

/**
 * A pane is a single square of the layout.
 * Panes have 1 or more tabs.
 * Each pane has 'drop zones' to control splitting when dragging tabs onto them.
 */
export class Pane {
  public tabs: Tab[] = []
  public element: HTMLElement
  public parent: Layout | null = null
  private tabBarElement!: HTMLElement
  private contentElement!: HTMLElement
  private addTabContainer!: HTMLElement
  private dropdown!: HTMLElement
  private isDropdownVisible: boolean = false
  private dropZones: Map<DropZone, HTMLElement> = new Map()
  private activeDropZone: DropZone | null = null

  constructor(container: HTMLElement) {
    this.element = container
    this.parent = null // Will be set when added to a layout
    this.render()
    this.setupEventListeners()
    this.setupDragAndDrop()
  }

  public addTab(tab: Tab): void {
    this.tabs.push(tab)
    if (this.tabs.length === 1) {
      tab.isActive = true
    }
    this.updateTabs()
  }

  private render(): void {
    this.element.className = 'pane'
    this.element.innerHTML = `
        <div class="tab-bar"></div>
        <div class="tab-content"></div>
        <div class="drop-zone drop-zone-left"></div>
        <div class="drop-zone drop-zone-right"></div>
        <div class="drop-zone drop-zone-top"></div>
        <div class="drop-zone drop-zone-bottom"></div>
    `

    this.tabBarElement = this.element.querySelector('.tab-bar') as HTMLElement
    this.contentElement = this.element.querySelector('.tab-content') as HTMLElement

    // Set up drop zones.
    this.dropZones.set(DropZone.LEFT, this.element.querySelector('.drop-zone-left') as HTMLElement)
    this.dropZones.set(DropZone.RIGHT, this.element.querySelector('.drop-zone-right') as HTMLElement)
    this.dropZones.set(DropZone.TOP, this.element.querySelector('.drop-zone-top') as HTMLElement)
    this.dropZones.set(DropZone.BOTTOM, this.element.querySelector('.drop-zone-bottom') as HTMLElement)

    // setup the "New Tab" button.
    this.addTabContainer = document.createElement('div')
    this.addTabContainer.className = 'add-tab-container'
    this.addTabContainer.innerHTML = `
        <div class="add-tab-btn">+</div>
        <div class="panel-type-dropdown">
            <div class="dropdown-item" data-type="${PanelType.LOGS}">${PanelType.LOGS}</div>
            <div class="dropdown-item" data-type="${PanelType.METRICS}">${PanelType.METRICS}</div>
            <div class="dropdown-item" data-type="${PanelType.MAP_VIEW}">${PanelType.MAP_VIEW}</div>
            <div class="dropdown-item" data-type="${PanelType.AGENT_DETAILS}">${PanelType.AGENT_DETAILS}</div>
        </div>
    `

    this.dropdown = this.addTabContainer.querySelector('.panel-type-dropdown') as HTMLElement
  }

  private setupEventListeners(): void {
    const addBtn = this.addTabContainer.querySelector('.add-tab-btn') as HTMLElement
    addBtn.addEventListener('click', (e) => {
      e.stopPropagation()
      this.toggleDropdown()
    })

    // Handle dropdown item clicks.
    const dropdownItems = this.addTabContainer.querySelectorAll('.dropdown-item')
    dropdownItems.forEach((item) => {
      item.addEventListener('click', (e) => {
        const panelType = (e.target as HTMLElement).getAttribute('data-type') as PanelType
        this.createNewTab(panelType)
        this.hideDropdown()
      })
    })

    // Close dropdown when clicking outside.
    document.addEventListener('click', (e) => {
      if (!this.addTabContainer.contains(e.target as Node)) {
        this.hideDropdown()
      }
    })
  }

  private setupDragAndDrop(): void {
    // Make this pane a drop target for tabs.
    this.tabBarElement.addEventListener('dragover', (e) => {
      e.preventDefault()
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'move'
      }
      this.setActiveDropZone(DropZone.TAB_BAR)
    })

    this.tabBarElement.addEventListener('dragleave', (e) => {
      if (!this.tabBarElement.contains(e.relatedTarget as Node)) {
        this.clearActiveDropZone(DropZone.TAB_BAR)
      }
    })

    this.tabBarElement.addEventListener('drop', (e) => {
      e.preventDefault()
      this.clearActiveDropZone(DropZone.TAB_BAR)

      const dragData = e.dataTransfer?.getData('text/plain')
      if (dragData) {
        const { sourceId, tabIndex } = JSON.parse(dragData)
        this.handleTabDrop(sourceId, tabIndex, DropZone.TAB_BAR)
      }
    })

    // Set up edge drop zones.
    this.dropZones.forEach((element, zone) => {
      element.addEventListener('dragover', (e) => {
        e.preventDefault()
        if (e.dataTransfer) {
          e.dataTransfer.dropEffect = 'move'
        }
        this.setActiveDropZone(zone)
      })

      element.addEventListener('dragenter', (e) => {
        e.preventDefault()
        if (e.dataTransfer) {
          e.dataTransfer.dropEffect = 'move'
        }
        this.setActiveDropZone(zone)
      })

      element.addEventListener('dragleave', (e) => {
        if (!element.contains(e.relatedTarget as Node)) {
          this.clearActiveDropZone(zone)
        }
      })

      element.addEventListener('drop', (e) => {
        e.preventDefault()
        this.clearActiveDropZone(zone)

        const dragData = e.dataTransfer?.getData('text/plain')
        if (dragData) {
          const { sourceId, tabIndex } = JSON.parse(dragData)
          this.handleTabDrop(sourceId, tabIndex, zone)
        }
      })
    })
  }

  private setActiveDropZone(zone: DropZone): void {
    this.clearAllDropZones()
    this.activeDropZone = zone

    if (zone === DropZone.TAB_BAR) {
      this.tabBarElement.classList.add('drag-over')
    } else {
      const element = this.dropZones.get(zone)
      if (element) {
        element.classList.add('active')
        this.element.classList.add(`split-preview-${zone}`)
      }
    }
  }

  private clearActiveDropZone(zone: DropZone): void {
    if (this.activeDropZone === zone) {
      this.clearAllDropZones()
    }
  }

  private clearAllDropZones(): void {
    this.activeDropZone = null
    this.tabBarElement.classList.remove('drag-over')
    this.dropZones.forEach((element) => element.classList.remove('active'))
    this.element.classList.remove(
      'split-preview-left',
      'split-preview-right',
      'split-preview-top',
      'split-preview-bottom'
    )
  }

  private handleTabDrop(sourceId: string, tabIndex: number, dropZone: DropZone): void {
    // Find the source pane.
    const allPanes = this.findAllPanes()
    const sourcePane = allPanes.find((pane) => pane.getPaneId() === sourceId)

    if (!sourcePane) return

    const draggedTab = sourcePane.tabs[tabIndex]
    if (!draggedTab) return

    if (dropZone === DropZone.TAB_BAR && sourcePane !== this) {
      // Move tab to existing pane.
      sourcePane.removeTab(tabIndex)
      this.addTab(draggedTab)
      this.activateTab(this.tabs.length - 1)
    } else if (dropZone !== DropZone.TAB_BAR) {
      // Perform split operation to create new pane.
      sourcePane.removeTab(tabIndex)
      this.performSplit(draggedTab, dropZone)
    }
  }

  private performSplit(draggedTab: Tab, dropZone: DropZone): void {
    // Find the parent layout that contains this pane.
    const parentLayout = this.parent
    if (!parentLayout) {
      return
    }

    // Mark all layouts in the hierarchy as being in a split operation.
    parentLayout.markSplitOperationInProgress(parentLayout, true)

    // Determine split direction based on drop zone.
    const isHorizontalSplit = dropZone === DropZone.LEFT || dropZone === DropZone.RIGHT
    const newDirection = isHorizontalSplit ? LayoutDirection.HORIZONTAL : LayoutDirection.VERTICAL
    const insertBefore = dropZone === DropZone.LEFT || dropZone === DropZone.TOP

    // Create new pane for the dragged tab.
    const newContainer = document.createElement('div')
    const newPane = new Pane(newContainer)
    newPane.addTab(draggedTab)

    // Find the index of this pane in the parent layout.
    const paneIndex = parentLayout.children.indexOf(this)
    if (paneIndex === -1) {
      parentLayout.markSplitOperationInProgress(parentLayout, false)
      return
    }

    // If parent layout has the same direction as our split, just insert the new pane.
    if (parentLayout.direction === newDirection) {
      const insertIndex = insertBefore ? paneIndex : paneIndex + 1
      parentLayout.insertChild(newPane, insertIndex)
    } else {
      // Create a new nested layout for the split.
      const nestedContainer = document.createElement('div')
      const nestedLayout = new Layout(nestedContainer, newDirection)

      // Remove this pane from parent and add it to nested layout.
      parentLayout.removeChild(this)

      if (insertBefore) {
        nestedLayout.addChild(newPane)
        nestedLayout.addChild(this)
      } else {
        nestedLayout.addChild(this)
        nestedLayout.addChild(newPane)
      }

      // Insert the nested layout at the original position.
      parentLayout.insertChild(nestedLayout, paneIndex)
    }

    // Clear the split operation flag and check for any needed simplification.
    parentLayout.markSplitOperationInProgress(parentLayout, false)
    parentLayout.checkSimplificationAfterSplit(parentLayout)
  }

  private toggleDropdown(): void {
    this.isDropdownVisible = !this.isDropdownVisible
    this.dropdown.classList.toggle('visible', this.isDropdownVisible)
  }

  private hideDropdown(): void {
    this.isDropdownVisible = false
    this.dropdown.classList.remove('visible')
  }

  private createNewTab(panelType: PanelType): void {
    const tabNumber = this.tabs.length + 1
    const content = this.generateContentForPanelType(panelType)
    const newTab = new Tab(`${panelType} ${tabNumber}`, content, panelType)
    this.addTab(newTab)
    this.activateTab(this.tabs.length - 1)
  }

  private generateContentForPanelType(panelType: PanelType): string {
    switch (panelType) {
      case PanelType.LOGS:
        return 'Log entries will appear here...\n\n[INFO] System started\n[DEBUG] Loading configuration\n[INFO] Ready to receive data'
      case PanelType.METRICS:
        return 'Performance metrics and statistics will be displayed here...\n\nCPU Usage: 45%\nMemory: 2.1GB / 8GB\nNetwork I/O: 1.2MB/s'
      case PanelType.MAP_VIEW:
        return 'Interactive map visualization will render here...\n\nMap dimensions: 100x100\nAgents: 5 active\nObstacles: 23'
      case PanelType.AGENT_DETAILS:
        return 'Agent status and details will be shown here...\n\nAgent ID: 001\nStatus: Active\nPosition: (45, 23)\nHealth: 100%'
      default:
        return 'Panel content will appear here...'
    }
  }

  public removeTab(index: number): void {
    if (index >= 0 && index < this.tabs.length) {
      this.tabs.splice(index, 1)
      // If we removed the active tab, activate another one.
      if (this.tabs.length > 0 && !this.tabs.some((tab) => tab.isActive)) {
        this.tabs[Math.min(index, this.tabs.length - 1)].isActive = true
      }
      this.updateTabs()

      // If this was the last tab in the pane, remove the pane.
      if (this.tabs.length === 0) {
        this.removePaneFromLayout()
      }
    }
  }

  public closeTab(index: number): void {
    if (index >= 0 && index < this.tabs.length) {
      this.removeTab(index)
    }
  }

  private removePaneFromLayout(): void {
    const parentLayout = this.parent
    if (!parentLayout) return

    parentLayout.removeChild(this)
  }

  public getPaneId(): string {
    if (!this.element.dataset.paneId) {
      this.element.dataset.paneId = `pane-${Math.random().toString(36).substr(2, 9)}`
    }
    return this.element.dataset.paneId
  }

  private findAllPanes(): Pane[] {
    const panes: Pane[] = []
    const containers = document.querySelectorAll('.pane')
    containers.forEach((container) => {
      if ((container as any).paneInstance) {
        panes.push((container as any).paneInstance)
      }
    })
    return panes
  }

  private enableGlobalDropZones(): void {
    const layoutContainer = document.querySelector('.layout-container')
    if (layoutContainer) {
      layoutContainer.classList.add('dragging')
    }
  }

  private disableGlobalDropZones(): void {
    const layoutContainer = document.querySelector('.layout-container')
    if (layoutContainer) {
      layoutContainer.classList.remove('dragging')
    }
  }

  private updateTabs(): void {
    // Store reference to this pane instance on the element.
    ;(this.element as any).paneInstance = this

    // Clear the entire tab bar.
    this.tabBarElement.innerHTML = ''

    // Add all tabs.
    this.tabs.forEach((tab, index) => {
      const tabElement = document.createElement('div')
      tabElement.className = `tab ${tab.isActive ? 'active' : ''}`
      tabElement.setAttribute('draggable', 'true')

      // Create tab content with title and close button.
      const tabTitle = document.createElement('span')
      tabTitle.className = 'tab-title'
      tabTitle.textContent = tab.title
      tabTitle.style.pointerEvents = 'none' // Critical: prevent this from interfering with drag

      const closeButton = document.createElement('span')
      closeButton.className = 'tab-close'
      closeButton.textContent = '×'
      closeButton.style.pointerEvents = 'auto' // Allow close button to be clickable
      closeButton.addEventListener('click', (e) => {
        e.stopPropagation()
        e.preventDefault()
        this.closeTab(index)
      })

      tabElement.appendChild(tabTitle)
      tabElement.appendChild(closeButton)

      // Add mousedown to help with drag initiation in Chrome
      tabElement.addEventListener('mousedown', (e) => {
        // Don't interfere with close button clicks
        if (e.target === closeButton) {
          return
        }
      })

      // Add drag event listeners.
      tabElement.addEventListener('dragstart', (e) => {
        const dragData = {
          sourceId: this.getPaneId(),
          tabIndex: index,
        }
        if (e.dataTransfer) {
          e.dataTransfer.setData('text/plain', JSON.stringify(dragData))
          e.dataTransfer.effectAllowed = 'move'
        }
        // Don't add dragging class immediately - it can cancel drag in Chrome
        setTimeout(() => {
          tabElement.classList.add('dragging')
        }, 10)
        this.enableGlobalDropZones()
      })

      tabElement.addEventListener('dragend', (e) => {
        tabElement.classList.remove('dragging')
        this.disableGlobalDropZones()
      })

      tabElement.addEventListener('click', () => this.activateTab(index))
      this.tabBarElement.appendChild(tabElement)
    })

    // Add the plus button after all tabs.
    this.tabBarElement.appendChild(this.addTabContainer)

    this.updateContent()
  }

  private activateTab(index: number): void {
    this.tabs.forEach((tab, i) => {
      tab.isActive = i === index
    })
    this.updateTabs()
  }

  private updateContent(): void {
    const activeTab = this.tabs.find((tab) => tab.isActive)
    this.contentElement.textContent = activeTab ? activeTab.content : ''
  }
}

// Layout child can be either a Pane or another Layout for nesting
export type LayoutChild = Pane | Layout

export class Layout {
  public container: HTMLElement
  public children: LayoutChild[] = []
  public parent: Layout | null = null
  private childContainers: HTMLElement[] = []
  private splitters: HTMLElement[] = []
  public direction: LayoutDirection
  private isDragging: boolean = false
  private dragSplitterIndex: number = -1
  private startPosition: number = 0
  private startSizes: number[] = []
  private isInSplitOperation: boolean = false

  constructor(container: HTMLElement, direction: LayoutDirection = LayoutDirection.HORIZONTAL) {
    this.container = container
    this.direction = direction
    this.parent = null // Root layout has no parent
    // Store reference to this layout instance.
    ;(this.container as any).layoutInstance = this
    this.render()
    this.setupSplitters()
  }

  private render(): void {
    // Only create the layout-container if it doesn't exist
    let layoutContainer = this.container.querySelector('.layout-container') as HTMLElement
    if (!layoutContainer) {
      this.container.innerHTML = `
        <div class="layout-container ${this.direction}">
        </div>
      `
    } else {
      // Update the direction class if container already exists
      layoutContainer.className = `layout-container ${this.direction}`
    }
  }

  public addChild(child: LayoutChild): void {
    this.children.push(child)
    child.parent = this
    this.updateLayout()
  }

  public insertChild(child: LayoutChild, index: number): void {
    this.children.splice(index, 0, child)
    child.parent = this
    this.updateLayout()
  }

  public removeChild(child: LayoutChild): void {
    const index = this.children.indexOf(child)
    if (index !== -1) {
      this.children.splice(index, 1)
      child.parent = null
      this.updateLayout()

      // Only check for simplification if we're not in the middle of a split operation.
      if (!this.isInSplitOperation && this.children.length <= 1) {
        this.checkForSimplification()
      }
    }
  }

  private checkForSimplification(): void {
    // If this layout is now empty or has only one child, we should simplify the layout tree.

    // Check if this is actually the root layout by looking for layout-container ID.
    const isRootLayout = this.container.id === 'layout-container'

    if (this.children.length === 1) {
      const singleChild = this.children[0]

      if (isRootLayout) {
        // For root layout, just replace all children with the single child.
        this.children = []
        this.addChild(singleChild)
      } else {
        // For non-root layouts, replace this layout with its single child.
        const parentLayout = this.parent
        if (parentLayout) {
          const layoutIndex = parentLayout.children.indexOf(this)
          if (layoutIndex !== -1) {
            // Remove single child from this layout first.
            this.children = []
            singleChild.parent = null
            // Replace this layout with the single child.
            parentLayout.removeChild(this)
            parentLayout.insertChild(singleChild, layoutIndex)
          }
        }
      }
    } else if (this.children.length === 0 && !isRootLayout) {
      const parentLayout = this.parent
      if (parentLayout) {
        parentLayout.removeChild(this)
      }
    }
  }

  private updateLayout(): void {
    const layoutContainer = this.container.querySelector('.layout-container') as HTMLElement
    if (!layoutContainer) {
      console.error('Layout container not found!')
      return
    }

    layoutContainer.innerHTML = ''
    this.childContainers = []
    this.splitters = []

    // Create containers for each child.
    this.children.forEach((child, index) => {
      const childContainer = document.createElement('div')
      childContainer.className = 'layout-child'
      childContainer.style.flex = '1'
      layoutContainer.appendChild(childContainer)
      this.childContainers.push(childContainer)

      // Set up the child in its container.
      if (child instanceof Pane) {
        // For Panes, append their existing element to the new container.
        childContainer.appendChild(child.element)
      } else {
        // For nested Layouts, set container and ensure proper rendering.
        child.setContainer(childContainer)
        child.render()
        child.updateLayout()
      }

      // Add splitter after each child except the last.
      if (index < this.children.length - 1) {
        const splitter = document.createElement('div')
        splitter.className = `splitter ${this.direction}`
        layoutContainer.appendChild(splitter)
        this.splitters.push(splitter)
      }
    })

    this.setupSplitters()
  }

  private setupSplitters(): void {
    this.splitters.forEach((splitter, index) => {
      splitter.addEventListener('mousedown', (e) => {
        this.isDragging = true
        this.dragSplitterIndex = index
        this.startPosition = this.direction === LayoutDirection.HORIZONTAL ? e.clientX : e.clientY
        this.startSizes = this.childContainers.map((container) =>
          this.direction === LayoutDirection.HORIZONTAL ? container.offsetWidth : container.offsetHeight
        )
        document.body.style.cursor = this.direction === LayoutDirection.HORIZONTAL ? 'col-resize' : 'row-resize'
        document.body.style.userSelect = 'none'
        e.preventDefault()
      })
    })

    document.addEventListener('mousemove', (e) => {
      if (!this.isDragging || this.dragSplitterIndex === -1) return

      const currentPosition = this.direction === LayoutDirection.HORIZONTAL ? e.clientX : e.clientY
      const delta = currentPosition - this.startPosition
      const containerSize =
        this.direction === LayoutDirection.HORIZONTAL ? this.container.offsetWidth : this.container.offsetHeight
      const splitterSize =
        this.direction === LayoutDirection.HORIZONTAL
          ? this.splitters[0]?.offsetWidth || 0
          : this.splitters[0]?.offsetHeight || 0

      const leftIndex = this.dragSplitterIndex
      const rightIndex = this.dragSplitterIndex + 1

      const newLeftSize = this.startSizes[leftIndex] + delta
      const newRightSize = this.startSizes[rightIndex] - delta
      const minSize = 100

      if (newLeftSize >= minSize && newRightSize >= minSize) {
        const totalFlexibleSize = containerSize - this.splitters.length * splitterSize

        // Calculate new sizes for all containers
        const newSizes = [...this.startSizes]
        newSizes[leftIndex] = newLeftSize
        newSizes[rightIndex] = newRightSize

        // Update flex values for all containers to maintain proper ratios
        newSizes.forEach((size, index) => {
          const flex = size / totalFlexibleSize
          this.childContainers[index].style.flex = `${flex}`
        })
      }
    })

    document.addEventListener('mouseup', () => {
      if (this.isDragging) {
        this.isDragging = false
        this.dragSplitterIndex = -1
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
    })
  }

  public setDirection(direction: LayoutDirection): void {
    this.direction = direction
    this.render()
    this.updateLayout()
  }

  public setContainer(container: HTMLElement): void {
    this.container = container
    // Store reference to this layout instance on the new container.
    ;(this.container as any).layoutInstance = this
  }

  public markSplitOperationInProgress(layout: Layout, inProgress: boolean): void {
    // Mark this layout and walk up the hierarchy.
    let current: Layout | null = layout
    while (current) {
      current.isInSplitOperation = inProgress
      current = current.parent
    }
  }

  public checkSimplificationAfterSplit(layout: Layout): void {
    // Check the entire hierarchy for needed simplification now that split is complete.
    let current: Layout | null = layout
    while (current) {
      if (current.children.length <= 1) {
        current.checkForSimplification()
      }
      current = current.parent
    }
  }
}

// Initialize the layout system with a basic horizontal split.
export function initLayout(): void {
  const container = document.getElementById('layout-container')
  if (!container) {
    console.error('Layout container not found')
    return
  }

  const layout = new Layout(container, LayoutDirection.HORIZONTAL)

  // Create initial panes.
  const leftContainer = document.createElement('div')
  const leftPane = new Pane(leftContainer)
  const leftTab = new Tab(
    'Left Panel',
    'This is the left side panel.\n\nYou can add more tabs using the + button.',
    PanelType.LOGS
  )
  leftPane.addTab(leftTab)

  const rightContainer = document.createElement('div')
  const rightPane = new Pane(rightContainer)
  const rightTab = new Tab(
    'Right Panel',
    'This is the right side panel.\n\nIt works independently from the left panel.',
    PanelType.MAP_VIEW
  )
  rightPane.addTab(rightTab)

  layout.addChild(leftPane)
  layout.addChild(rightPane)
}

// Auto-initialize when the DOM is ready.
document.addEventListener('DOMContentLoaded', initLayout)
