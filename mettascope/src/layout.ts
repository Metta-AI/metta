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

  constructor(container: HTMLElement) {
    this.element = container
    this.render()
    this.setupEventListeners()
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
    `

    this.tabBarElement = this.element.querySelector('.tab-bar') as HTMLElement
    this.contentElement = this.element.querySelector('.tab-content') as HTMLElement

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

  private updateTabs(): void {
    // Clear the entire tab bar.
    this.tabBarElement.innerHTML = ''

    // Add all tabs.
    this.tabs.forEach((tab, index) => {
      const tabElement = document.createElement('div')
      tabElement.className = `tab ${tab.isActive ? 'active' : ''}`
      tabElement.textContent = tab.title
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
    this.container = container
    this.direction = direction
    this.render()
    this.setupSplitters()
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

  public removeChild(child: LayoutChild): void {
    const index = this.children.indexOf(child)
    if (index !== -1) {
      this.children.splice(index, 1)
      this.updateLayout()
    }
  }

  private updateLayout(): void {
    const layoutContainer = this.container.querySelector('.layout-container') as HTMLElement
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
        // For nested Layouts, update their container
        child.container = childContainer
        child.render()
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
