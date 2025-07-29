import { Vec2f, Mat3f } from './vector_math.js'

// Custom layout system for managing tabs and panes.

export enum PanelType {
  LOGS = 'Logs',
  METRICS = 'Metrics',
  MAP_VIEW = 'Map View',
  AGENT_DETAILS = 'Agent Details'
}

export enum LayoutDirection {
  HORIZONTAL = 'horizontal',
  VERTICAL = 'vertical'
}

export enum DropZone {
  TAB_BAR = 'tab-bar',
  LEFT = 'left',
  RIGHT = 'right',
  TOP = 'top',
  BOTTOM = 'bottom'
}

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

export class Pane {
  public tabs: Tab[] = []
  public element: HTMLElement
  private tabBarElement!: HTMLElement
  private contentElement!: HTMLElement
  private addTabContainer!: HTMLElement
  private dropdown!: HTMLElement
  private isDropdownVisible: boolean = false
  private isDragTarget: boolean = false
  private dropZones: Map<DropZone, HTMLElement> = new Map()
  private activeDropZone: DropZone | null = null

  constructor(container: HTMLElement) {
    this.element = container
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

            // Set up drop zones
    this.dropZones.set(DropZone.LEFT, this.element.querySelector('.drop-zone-left') as HTMLElement)
    this.dropZones.set(DropZone.RIGHT, this.element.querySelector('.drop-zone-right') as HTMLElement)
    this.dropZones.set(DropZone.TOP, this.element.querySelector('.drop-zone-top') as HTMLElement)
    this.dropZones.set(DropZone.BOTTOM, this.element.querySelector('.drop-zone-bottom') as HTMLElement)

    // Create the add-tab-container separately.
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
    dropdownItems.forEach(item => {
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
    // Make this pane a drop target for tabs
    this.tabBarElement.addEventListener('dragover', (e) => {
      e.preventDefault()
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

                    // Set up edge drop zones
    this.dropZones.forEach((element, zone) => {
            element.addEventListener('dragover', (e) => {
        e.preventDefault()
        this.setActiveDropZone(zone)
      })

      element.addEventListener('dragenter', (e) => {
        e.preventDefault()
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
    this.dropZones.forEach(element => element.classList.remove('active'))
    this.element.classList.remove('split-preview-left', 'split-preview-right', 'split-preview-top', 'split-preview-bottom')
  }

      private handleTabDrop(sourceId: string, tabIndex: number, dropZone: DropZone): void {
    // Find the source pane
    const allPanes = this.findAllPanes()
    const sourcePane = allPanes.find(pane => pane.getPaneId() === sourceId)

    if (!sourcePane) return

    const draggedTab = sourcePane.tabs[tabIndex]
    if (!draggedTab) return

    if (dropZone === DropZone.TAB_BAR && sourcePane !== this) {
      // Simple tab move to existing pane
      sourcePane.removeTab(tabIndex)
      this.addTab(draggedTab)
      this.activateTab(this.tabs.length - 1)
    } else if (dropZone !== DropZone.TAB_BAR) {
      // Split operation
      sourcePane.removeTab(tabIndex)
      this.performSplit(draggedTab, dropZone)
    }
  }

  private performSplit(draggedTab: Tab, dropZone: DropZone): void {
    // Find the parent layout that contains this pane
    const parentLayout = this.findParentLayout()
    if (!parentLayout) return

    // Determine split direction
    const isHorizontalSplit = dropZone === DropZone.LEFT || dropZone === DropZone.RIGHT
    const newDirection = isHorizontalSplit ? LayoutDirection.HORIZONTAL : LayoutDirection.VERTICAL
    const insertBefore = dropZone === DropZone.LEFT || dropZone === DropZone.TOP

    // Create new pane for the dragged tab
    const newContainer = document.createElement('div')
    const newPane = new Pane(newContainer)
    newPane.addTab(draggedTab)

    // Find the index of this pane in the parent layout
    const paneIndex = parentLayout.getChildren().indexOf(this)
    if (paneIndex === -1) return

    // If parent layout has the same direction as our split, just insert the new pane
    if (parentLayout.getDirection() === newDirection) {
      const insertIndex = insertBefore ? paneIndex : paneIndex + 1
      parentLayout.insertChild(newPane, insertIndex)
    } else {
      // Create a new nested layout for the split
      const nestedContainer = document.createElement('div')
      const nestedLayout = new Layout(nestedContainer, newDirection)

      // Remove this pane from parent and add it to nested layout
      parentLayout.removeChild(this)

      if (insertBefore) {
        nestedLayout.addChild(newPane)
        nestedLayout.addChild(this)
      } else {
        nestedLayout.addChild(this)
        nestedLayout.addChild(newPane)
      }

      // Insert the nested layout at the original position
      parentLayout.insertChild(nestedLayout, paneIndex)
    }
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
      // If we removed the active tab, activate another one
      if (this.tabs.length > 0 && !this.tabs.some(tab => tab.isActive)) {
        this.tabs[Math.min(index, this.tabs.length - 1)].isActive = true
      }
      this.updateTabs()
    }
  }

      public closeTab(index: number): void {
    if (index >= 0 && index < this.tabs.length) {
      this.removeTab(index)

      // If this was the last tab in the pane, remove the pane
      if (this.tabs.length === 0) {
        this.removePaneFromLayout()
      }
    }
  }

            private removePaneFromLayout(): void {
    const parentLayout = this.findParentLayout()
    if (!parentLayout) return

    // Remove this pane from the parent layout
    parentLayout.removeChild(this)
  }





  private findParentLayoutOf(targetLayout: Layout): Layout | null {
    // Walk up the DOM to find the parent of the target layout
    let current = targetLayout.getContainer().parentElement
    while (current) {
      if ((current as any).layoutInstance && (current as any).layoutInstance !== targetLayout) {
        return (current as any).layoutInstance
      }
      current = current.parentElement
    }
    return null
  }

  public getPaneId(): string {
    if (!this.element.dataset.paneId) {
      this.element.dataset.paneId = `pane-${Math.random().toString(36).substr(2, 9)}`
    }
    return this.element.dataset.paneId
  }

  private findAllPanes(): Pane[] {
    // This is a simplified approach - in a real implementation,
    // you might want to traverse the layout tree more systematically
    const panes: Pane[] = []
    const containers = document.querySelectorAll('.pane')
    containers.forEach(container => {
      if ((container as any).paneInstance) {
        panes.push((container as any).paneInstance)
      }
    })
    return panes
  }

    private findParentLayout(): Layout | null {
    // Walk up the DOM to find the immediate parent layout
    let current = this.element.parentElement

    while (current) {
      // Check if this element has a layout instance
      if ((current as any).layoutInstance) {
        return (current as any).layoutInstance
      }

      // Also check parent element in case layout instance is stored there
      if (current.parentElement && (current.parentElement as any).layoutInstance) {
        return (current.parentElement as any).layoutInstance
      }

      current = current.parentElement
    }

    return null
  }

  private enableGlobalDropZones(): void {
    // Find the root layout container and add dragging class
    const layoutContainer = document.querySelector('.layout-container')
    if (layoutContainer) {
      layoutContainer.classList.add('dragging')
    }
  }

  private disableGlobalDropZones(): void {
    // Find the root layout container and remove dragging class
    const layoutContainer = document.querySelector('.layout-container')
    if (layoutContainer) {
      layoutContainer.classList.remove('dragging')
    }
  }

  private updateTabs(): void {
    // Store reference to this pane instance on the element
    (this.element as any).paneInstance = this

    // Clear the entire tab bar.
    this.tabBarElement.innerHTML = ''

        // Add all tabs.
    this.tabs.forEach((tab, index) => {
      const tabElement = document.createElement('div')
      tabElement.className = `tab ${tab.isActive ? 'active' : ''}`
      tabElement.draggable = true

      // Create tab content with title and close button
      const tabTitle = document.createElement('span')
      tabTitle.className = 'tab-title'
      tabTitle.textContent = tab.title

      const closeButton = document.createElement('span')
      closeButton.className = 'tab-close'
      closeButton.textContent = '×'
      closeButton.addEventListener('click', (e) => {
        e.stopPropagation() // Prevent tab activation
        this.closeTab(index)
      })

      tabElement.appendChild(tabTitle)
      tabElement.appendChild(closeButton)

            // Add drag event listeners
      tabElement.addEventListener('dragstart', (e) => {
        const dragData = {
          sourceId: this.getPaneId(),
          tabIndex: index
        }
        e.dataTransfer?.setData('text/plain', JSON.stringify(dragData))
        tabElement.classList.add('dragging')
        // Enable drop zones globally
        this.enableGlobalDropZones()
      })

      tabElement.addEventListener('dragend', () => {
        tabElement.classList.remove('dragging')
        // Disable drop zones globally
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
  private container: HTMLElement
  private children: LayoutChild[] = []
  private childContainers: HTMLElement[] = []
  private splitters: HTMLElement[] = []
  private direction: LayoutDirection
  private isDragging: boolean = false
  private dragSplitterIndex: number = -1
  private startPosition: number = 0
  private startSizes: number[] = []

  constructor(container: HTMLElement, direction: LayoutDirection = LayoutDirection.HORIZONTAL) {
    this.container = container;
    this.direction = direction;
    // Store reference to this layout instance
    (this.container as any).layoutInstance = this;
    this.render();
    this.setupSplitters();
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="layout-container ${this.direction}">
      </div>
    `
  }

  public addChild(child: LayoutChild): void {
    this.children.push(child)
    this.updateLayout()
  }

  public insertChild(child: LayoutChild, index: number): void {
    this.children.splice(index, 0, child)
    this.updateLayout()
  }

        public removeChild(child: LayoutChild): void {
    const index = this.children.indexOf(child)
    if (index !== -1) {
      this.children.splice(index, 1)
      this.updateLayout()

      // If this layout is now empty or has only one child, check if we need cleanup
      if (this.children.length <= 1) {
        this.checkForSimplification()
      }
    }
  }

        private checkForSimplification(): void {
    // Check if this is actually the root layout by looking for layout-container ID
    const isRootLayout = this.container.id === 'layout-container'

    if (this.children.length === 1) {
      const singleChild = this.children[0]

      if (isRootLayout) {
        // For root layout, just replace all children with the single child
        this.children = []
        this.addChild(singleChild)
      } else {
        // For non-root layouts, replace this layout with its single child
        const parentLayout = this.findParentLayoutAggressively()
        if (parentLayout) {
          const layoutIndex = parentLayout.getChildren().indexOf(this)
          if (layoutIndex !== -1) {
            // Remove single child from this layout first
            this.children = []
            // Replace this layout with the single child
            parentLayout.removeChild(this)
            parentLayout.insertChild(singleChild, layoutIndex)
          }
        }
      }
    } else if (this.children.length === 0 && !isRootLayout) {
      const parentLayout = this.findParentLayoutAggressively()
      if (parentLayout) {
        parentLayout.removeChild(this)
      }
    }
  }

    private findParentLayoutAggressively(): Layout | null {
    // Try multiple strategies to find the parent
    let current = this.container.parentElement
    let depth = 0

    while (current && depth < 10) {
      if ((current as any).layoutInstance && (current as any).layoutInstance !== this) {
        return (current as any).layoutInstance
      }

      // Also check if this element has a layout-container child that might have the instance
      const layoutContainer = current.querySelector('.layout-container')
      if (layoutContainer && (layoutContainer as any).layoutInstance && (layoutContainer as any).layoutInstance !== this) {
        return (layoutContainer as any).layoutInstance
      }

      current = current.parentElement
      depth++
    }

    return null
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

    // Create containers for each child
    this.children.forEach((child, index) => {
      // Create child container
      const childContainer = document.createElement('div')
      childContainer.className = 'layout-child'
      childContainer.style.flex = '1'
      layoutContainer.appendChild(childContainer)
      this.childContainers.push(childContainer)

      // Set up the child in its container
      if (child instanceof Pane) {
        // For Panes, append their existing element to the new container
        childContainer.appendChild(child.element)
      } else {
        // For nested Layouts, update their container and re-render
        child.setContainer(childContainer)
        child.render()
        child.updateLayout()
      }

      // Add splitter after each child except the last
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
        this.startSizes = this.childContainers.map(container =>
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
      const containerSize = this.direction === LayoutDirection.HORIZONTAL
        ? this.container.offsetWidth
        : this.container.offsetHeight
      const splitterSize = this.direction === LayoutDirection.HORIZONTAL
        ? this.splitters[0]?.offsetWidth || 0
        : this.splitters[0]?.offsetHeight || 0

      const leftIndex = this.dragSplitterIndex
      const rightIndex = this.dragSplitterIndex + 1

      const newLeftSize = this.startSizes[leftIndex] + delta
      const newRightSize = this.startSizes[rightIndex] - delta
      const minSize = 100

      if (newLeftSize >= minSize && newRightSize >= minSize) {
        const totalFlexibleSize = containerSize - (this.splitters.length * splitterSize)
        const leftFlex = newLeftSize / totalFlexibleSize
        const rightFlex = newRightSize / totalFlexibleSize

        this.childContainers[leftIndex].style.flex = `${leftFlex}`
        this.childContainers[rightIndex].style.flex = `${rightFlex}`
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

  public getChildren(): LayoutChild[] {
    return this.children
  }

  public getDirection(): LayoutDirection {
    return this.direction
  }

  public setDirection(direction: LayoutDirection): void {
    this.direction = direction
    this.render()
    this.updateLayout()
  }

  public getContainer(): HTMLElement {
    return this.container
  }

  public setContainer(container: HTMLElement): void {
    this.container = container;
    // Store reference to this layout instance on the new container
    (this.container as any).layoutInstance = this;
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

  // Create initial panes
  const leftContainer = document.createElement('div')
  const leftPane = new Pane(leftContainer)
  const leftTab = new Tab('Left Panel', 'This is the left side panel.\n\nYou can add more tabs using the + button.', PanelType.LOGS)
  leftPane.addTab(leftTab)

  const rightContainer = document.createElement('div')
  const rightPane = new Pane(rightContainer)
  const rightTab = new Tab('Right Panel', 'This is the right side panel.\n\nIt works independently from the left panel.', PanelType.MAP_VIEW)
  rightPane.addTab(rightTab)

  layout.addChild(leftPane)
  layout.addChild(rightPane)
}

// Auto-initialize when the DOM is ready.
document.addEventListener('DOMContentLoaded', initLayout)
